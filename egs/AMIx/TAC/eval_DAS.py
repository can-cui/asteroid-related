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
from speechbrain.lobes.beamform_multimic import DelaySum_Beamformer
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe.wpe import wpe

beamformer = DelaySum_Beamformer()


def beamforming(signal):
    batch = signal.permute(0, 2, 1)
    beamformed_array1 = beamformer(batch)
    beamformed_array1 = beamformed_array1.permute(0, 2, 1)
    return beamformed_array1


parser = argparse.ArgumentParser()

parser.add_argument("--test_json", type=str, required=True, help="Test json file")
parser.add_argument("--spk_dict", type=str, required=True, help="spk dict")
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)
parser.add_argument("--n_src", default=1, help="Number of sources")
parser.add_argument("--max_mics", type=int, default=2, help="Number of sources")

compute_metrics = ["si_sdr"]  # , "sdr", "sir", "sar", "stoi"]
# compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    # model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    # model = FasNetTAC.from_pretrained(model_path)
    # Handle device placement
    # if conf["use_gpu"]:
    #     model.cuda()
    # model_device = next(model.parameters()).device
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
    # print("ex_save_dir")
    # print(ex_save_dir)
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.no_grad().__enter__()
    stft_options = dict(size=512, shift=128)
    for idx in tqdm(range(len(test_set))):
        # for idx in tqdm(range(10)):

        # Forward the network on the mixture.
        mix, sources, valid_mics = tensors_to_device(test_set[idx], device=model_device)
        valid_mics = torch.tensor([valid_mics]).to(sources.device)
        # print(mix[None].shape)  # torch.Size([1, 8, 64000])
        # print(sources[None].shape)  # torch.Size([1, 8, 1, 64000])
        ## WPE
        Y = stft(mix[None].squeeze(0).cpu(), **stft_options)
        Y = Y.transpose(2, 0, 1)
        Z = wpe(Y)
        sig = istft(
            Z.transpose(1, 2, 0),
            size=stft_options["size"],
            shift=stft_options["shift"],
        )
        sig = torch.from_numpy(sig).to(sources.device)
        # print(sig[None].shape)
        # est_sources = beamforming(sig[None])
        # ## WPE
        Y = stft(sources[None].squeeze(2).squeeze(0).cpu(), **stft_options)
        Y = Y.transpose(2, 0, 1)
        Z = wpe(Y)
        sources = istft(
            Z.transpose(1, 2, 0),
            size=stft_options["size"],
            shift=stft_options["shift"],
        )
        sources = torch.from_numpy(sources).to(valid_mics.device)

        # sources = beamforming(sources[None])
        # loss, reordered_sources = loss_func(est_sources, sources[None][:, 0], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources[0].squeeze(0).cpu().data.numpy()
        # est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
        ### apply DAS to source
        # est_sources_np =
        utt_metrics = get_metrics(
            sig[0].cpu(),
            sources_np,
            sig[0].cpu(),
            sample_rate=conf["sample_rate"],
            metrics_list=compute_metrics,
        )
        # utt_metrics["mix_path"] = test_set.examples[idx]["1"]["mixture"]
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        # if idx in save_idx:
        #     local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
        #     os.makedirs(local_save_dir, exist_ok=True)
        #     for chn, mix_chn in enumerate(mix_np):
        #         sf.write(
        #             local_save_dir + "mixture{}.wav".format(chn + 1),
        #             mix_chn,
        #             conf["sample_rate"],
        #         )
        #     # Loop over the sources and estimates
        #     for src_idx, src in enumerate(sources_np):
        #         sf.write(local_save_dir + "s{}.wav".format(src_idx + 1), src, conf["sample_rate"])
        #     for src_idx, est_src in enumerate(est_sources_np):
        #         # Normalise est src with the absolute value of mix
        #         normalized_est_src = np.array(
        #             [(est_src / np.max(np.abs(est_src))) * max(abs(mix_np[0]))], np.float32
        #         ).squeeze(0)
        #         sf.write(
        #             local_save_dir + "s{}_estimate.wav".format(src_idx + 1),
        #             normalized_est_src,
        #             conf["sample_rate"],
        #         )
        #     # Write local metrics to the example folder.
        #     with open(local_save_dir + "metrics.json", "w") as f:
        #         json.dump(utt_metrics, f, indent=0)

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
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    # conf_path = os.path.join("local", "conf.yml")
    # with open(conf_path) as f:
    #     train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = 16000
    # arg_dic["train_conf"] = train_conf
    main(arg_dic)
