import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pprint import pprint
import sys
from asteroid.models.fasnet import FasNetTAC
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from local.amix_dataset_dm import AMIxDataset
from asteroid.utils import tensors_to_device
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml


class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        mix, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        speaker_directory, _ = batch.speaker_directory

        valid_mics = batch.valid_mics
        mixture = mix.permute(0, 2, 1)
        # if freeze_sep:  # freeze separation params to train ASR
        # with torch.no_grad():
        est_src = self.modules.Fasnet(mixture, valid_mics)
        # else:
        #     est_src = self.modules.Fasnet(mixture, valid_mics)

        wavs = est_src.squeeze(1)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        feats = self.hparams.compute_features_mel(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # if stage == sb.Stage.TRAIN:
        #     if hasattr(self.hparams, "augmentation"):
        #         feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)

        if current_epoch <= self.hparams.asr_apochs:
            speaker_directory = torch.zeros(
                speaker_directory.size(0),
                speaker_directory.size(1),
                speaker_directory.size(2),
                device=self.device,
            )
            spk_emb = torch.zeros(src.size(0), src.size(1), 192, device=self.device)
        else:
            spk_emb = self.modules.embedding_model(feats, wav_lens)

        enc_out, pred, pred_spk = self.modules.Transformer(
            src,
            tokens_bos,
            spk_emb,
            speaker_directory,
            wav_lens,
            pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        hyps_spk = None
        if stage == sb.Stage.TRAIN:
            hyps = None
            hyps_spk = None
        elif stage == sb.Stage.VALID:
            hyps = None
            hyps_spk = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, hyps_spk, _ = self.hparams.valid_search(
                    enc_out.detach(),
                    wav_lens,
                    spk_emb.detach(),
                    speaker_directory.detach(),
                )
        elif stage == sb.Stage.TEST:
            hyps, hyps_spk, _ = self.hparams.test_search(
                enc_out.detach(),
                wav_lens,
                spk_emb.detach(),
                speaker_directory.detach(),
            )
            # print(hyps.shape)

        return (p_ctc, p_seq, wav_lens, hyps, pred_spk, hyps_spk)

    def get_sent_spk(self, asr_all, spk_all):
        pad_token = 0
        cs_token = 3
        all_sent_spk_list = []
        for i in range(len(asr_all)):
            try:
                asr = asr_all[i].tolist()
            except:
                asr = asr_all[i]
            if pad_token in asr:
                asr = asr[: asr.index(pad_token)]
            try:
                spk = spk_all[i].tolist()
            except:
                spk = spk_all[i]
            if pad_token in spk:
                spk = spk[: spk.index(pad_token)]
            if not len(spk) == 0:
                try:
                    sc_index = [i for i, x in enumerate(asr) if x == cs_token] + [len(asr)]
                    begin = 0
                    sent_spk_list = []
                    for ind in sc_index:
                        sent_spk = spk[begin:ind]
                        spk_id = max(sent_spk, key=sent_spk.count)
                        sent_spk_list.append(spk_id)
                        begin = ind
                        if begin == sc_index[-1]:
                            exit
                except Exception as e:
                    print(e)
                    print(spk)
            else:
                # except Exception as e:
                sent_spk_list = []
            all_sent_spk_list.append(sent_spk_list)

        return all_sent_spk_list

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps, pred_spk, hyps_spk) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        speaker_label, spk_lens = batch.speaker_label

        mix, wav_lens = batch.sig
        mix = mix[:, :, 0]  # take the first channel
        # print("mix.shape")
        # print(mix.shape) # torch.Size([2, 411464])
        # src, _ = batch.src  # torch.Size([24, 407039, 2, 1])
        # src = src[:, :, 0]  # take the first channel
        # print("src.shape")
        # print(src.shape)  # torch.Size([2, 411464, 1])
        # print("est_src.shape")
        # print(est_src.shape)  # torch.Size([2, 411464, 1])
        # loss_sep = self.hparams.sisdr_cost(src, est_src).sum()

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat([tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens).sum()

        # now as training progresses we use real prediction from the prev step instead of teacher forcing

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens).sum()

        loss_asr = self.hparams.ctc_weight * loss_ctc + (1 - self.hparams.ctc_weight) * loss_seq

        loss_spk = self.hparams.spk_loss(pred_spk, speaker_label, length=spk_lens).sum()

        current_epoch = self.hparams.epoch_counter.current
        # if current_epoch <= self.hparams.asr_apochs:  # epochs for retrain ASR only
        #     loss = loss_asr
        # else:
        #     loss = (self.hparams.spk_weight * loss_spk +
        #             (1 - self.hparams.spk_weight) * loss_asr)

        if current_epoch <= self.hparams.asr_apochs:  # epochs for retrain ASR only
            loss = loss_asr
        else:  # epochs spk
            loss = (
                self.hparams.spk_loss_weight * loss_spk
                + (1 - self.hparams.spk_loss_weight) * loss_asr
            )
        # elif current_epoch > self.hparams.freeze_sep_epochs:  # epochs for joint ASR & spk & sep
        #     loss = (self.hparams.sep_loss_weight * loss_sep +
        #             self.hparams.spk_loss_weight * loss_spk +
        #             (1 - self.hparams.sep_loss_weight -
        #              self.hparams.spk_loss_weight) * loss_asr)

        # mixture_signal = torch.stack([mix] * 1, dim=-1)  # 1 spk
        # mixture_signal = mixture_signal.to(src.device)
        # print("mixture_signal.shape")
        # print(mixture_signal.shape)
        # print("src.shape")
        # print(src.shape)
        # sisnr_baseline = self.hparams.sisdr_cost(src, mixture_signal).sum()

        # if current_epoch == 1:  # test pretrained separation performance
        #     if hasattr(self.hparams, "n_audio_to_save"):
        #         if self.hparams.n_audio_to_save > 0:
        #             self.save_audio(ids[0], mix, src, est_src, "_pretrained")
        #             self.save_results(ids[0], mixture_signal, src, est_src,
        #                               loss_sep, sisnr_baseline, "_pretrained")
        #             self.hparams.n_audio_to_save += -1

        if stage != sb.Stage.TRAIN:
            # current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (stage == sb.Stage.TEST):
                # Decode token terms to words
                predicted_words = [tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps]
                target_words = [wrd.split(" ") for wrd in batch.wrd_spk]
                print("target_words")
                print(target_words)
                print("predicted_words")
                print(predicted_words)
                self.wer_metric.append(ids, predicted_words, target_words)
                # if hasattr(self.hparams, "n_audio_to_save"):
                #     if self.hparams.n_audio_to_save > 0:
                #         self.save_audio(ids[0], mix, src, est_src, "_final")
                #         self.save_results(ids[0], mixture_signal, src, est_src,
                #                           loss_sep, sisnr_baseline, "_final")
                #         self.hparams.n_audio_to_save += -1
                predicted_spk = [spk_seq for spk_seq in hyps_spk]
                target_spk = [spk_seq for spk_seq in speaker_label]
                print("target_spk")
                print(target_spk)
                print("predicted_spk")
                print(predicted_spk)
                tgt_spk_sent = self.get_sent_spk(tokens, speaker_label)

                pred_spk_sent = self.get_sent_spk(hyps, hyps_spk)
                print("tgt_spk_sent")
                print(tgt_spk_sent)
                print("pred_spk_sent")
                print(pred_spk_sent)
                # exit()
                self.ser_metric.append(ids, pred_spk_sent, tgt_spk_sent)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
            self.acc_metric_spk.append(pred_spk, speaker_label, spk_lens)

        if not torch.isfinite(loss):
            wavs, wav_lens = batch.sig
            # print("wavs.shape")
            # print(wavs.shape)  # torch.Size([2, 584138])
            save_path = os.path.join(self.hparams.save_folder, "nan_loss_audio")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            for idx in range(len(ids)):
                save_file = os.path.join(save_path, "{}.wav".format(ids[idx]))
                torchaudio.save(
                    save_file,
                    wavs[idx].unsqueeze(0).cpu(),
                    self.hparams.sample_rate,
                )
                save_file_text = os.path.join(save_path, "{}_gold.txt".format(ids[idx]))
                with open(save_file_text, "w") as file:
                    file.write(batch.wrd_spk[idx])

        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.acc_metric_spk = self.hparams.acc_computer()
            self.ser_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["ACC_SPK"] = self.acc_metric_spk.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["SER"] = self.ser_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={
                    "ACC": stage_stats["ACC"],
                    "ACC_SPK": stage_stats["ACC_SPK"],
                    "epoch": epoch,
                },
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "ACC_SPK": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(loss / self.grad_accumulation_factor).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)
        else:
            if self.bfloat16_mix_prec:
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()


# parser = argparse.ArgumentParser()

# parser.add_argument("--test_json", type=str, required=True, help="Test json file")
# parser.add_argument("--spk_dict", type=str, required=True, help="spk dict")
# parser.add_argument(
#     "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
# )
# parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
# parser.add_argument(
#     "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
# )
# parser.add_argument("--n_src", default=1, help="Number of sources")
# parser.add_argument("--max_mics", type=int, default=2, help="Number of sources")

compute_metrics = ["si_sdr"]  # , "sdr", "sir", "sar", "stoi"]
# compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf, model):
    # model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    # model = FasNetTAC.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = AMIxDataset(
        conf["test_json"],
        conf["spk_dict"],
        n_src=conf["n_src"],
        max_mics=conf["max_mics"],
        train=False,
    )

    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["exp_dir"], "examples/")
    print("ex_save_dir")
    print(ex_save_dir)
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # for idx in tqdm(range(10)):

        # Forward the network on the mixture.
        mix, sources, valid_mics = tensors_to_device(test_set[idx], device=model_device)
        valid_mics = torch.tensor([valid_mics]).to(sources.device)
        est_sources = model(mix[None], valid_mics[None])
        loss, reordered_sources = loss_func(est_sources, sources[None][:, 0], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources[0].cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        utt_metrics = get_metrics(
            mix_np[0],
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=compute_metrics,
        )
        # utt_metrics["mix_path"] = test_set.examples[idx]["1"]["mixture"]
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            for chn, mix_chn in enumerate(mix_np):
                sf.write(
                    local_save_dir + "mixture{}.wav".format(chn + 1),
                    mix_chn,
                    conf["sample_rate"],
                )
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx + 1), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np):
                # Normalise est src with the absolute value of mix
                normalized_est_src = np.array(
                    [(est_src / np.max(np.abs(est_src))) * max(abs(mix_np[0]))], np.float32
                ).squeeze(0)
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx + 1),
                    normalized_est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(conf["exp_dir"], "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    print("Overall metrics :")
    pprint(final_results)
    with open(os.path.join(conf["exp_dir"], "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    # from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # model = hparams["model"]
    model = asr_brain.modules.Fasnet

    # args = parser.parse_args()
    # arg_dic = dict(vars(args))

    # Load training config
    # conf_path = os.path.join(args.exp_dir, "conf.yml")
    # with open(conf_path) as f:
    #     train_conf = yaml.safe_load(f)
    # arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    # arg_dic["train_conf"] = train_conf
    main(hparams, model)
