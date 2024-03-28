import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import numpy as np
# from asteroid.models.fasnet import FasNetTAC
from model.trans_fasnet import TransFasNetTAC

from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from local.tac_dataset import TACDataset
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from data.data_loader_dynamicMix import SpectrogramDataset

parser = argparse.ArgumentParser()

# parser.add_argument("--test_json", type=str, required=True, help="Test json file")
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory in exp_dir where the eval results" " will be stored",
)

parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)
parser.add_argument("--n_src", default=2, help="Number of sources")
parser.add_argument("--n_ch", type=int, default=2, help="Number of sources")

# compute_metrics = ["si_sdr" , "sdr", "sir", "sar", "stoi"]
# 
compute_metrics = ["si_sdr"]

def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    # model_path = os.path.join(conf["exp_dir"], "checkpoints","epoch=18-step=314811.ckpt")
    model = TransFasNetTAC.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    # test_set = TACDataset(args.test_json, train=False)
    test_set = SpectrogramDataset(conf["data_path"],
                                manifest_filepath_list=conf["test_dir"],
                                num_spk=conf["n_src"],
                                max_mics=conf["n_ch"],
                                segment=None,
                                normalize=True,
                                augment=False)

    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["exp_dir"], "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):

        # Forward the network on the mixture.
        mix, sources, valid_mics = tensors_to_device(test_set[idx], device=model_device)
        valid_mics = torch.tensor([valid_mics]).to(sources.device)
        est_sources = model(mix[None], valid_mics[None])
        loss, reordered_sources = loss_func(est_sources, sources[None][:, 0], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources[0].cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # print(mix_np[0].shape) # (159160,)
        # print(sources_np.shape) #(2, 159160)
        # print(est_sources_np.shape) #(2, 159160)
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
            mix_np[0] = mix_np[0] / np.max(np.abs(mix_np[0]))
            sf.write(local_save_dir + "mixture.wav", mix_np[0], conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                src = src / np.max(np.abs(src))
                sf.write(local_save_dir + "s{}.wav".format(src_idx + 1), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np):
                est_src = est_src / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx + 1),
                    est_src,
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
    model_dict = torch.load(model_path, map_location="cpu")

    # publishable = save_publishable(
    #     os.path.join(conf["exp_dir"], "publish_dir"),
    #     model_dict,
    #     metrics=final_results,
    #     train_conf=train_conf,
    # )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["data_path"] = train_conf["data"]["data_path"]
    arg_dic["train_conf"] = train_conf
    arg_dic["test_dir"] = train_conf["data"]["test_dir"]
    arg_dic["n_src"] = train_conf["net"]["n_src"]

    main(arg_dic)
