import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# from local.amix_dataset import AMIxDataset
from local.amix_dataset_dm import AMIxDataset
from asteroid.engine.optimizers import make_optimizer

# from asteroid.engine.system_TAC_snr import System
from asteroid.engine.system import System

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import ConvTasNet_mvdr
from asteroid.metrics import get_metrics

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py.
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
parser.add_argument("--max_mics", type=int, default=2, help="Number of sources")

compute_metrics = ["si_sdr"]
# compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


class TACSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, valid_channels = batch
        est_targets = self.model(inputs, valid_channels)

        loss = self.loss_func(est_targets, targets[:, 0]).mean()  # first channel is used as ref
        # print("*" * 100)
        # print(batch)
        # # ref: tensor([3, 2], device='cuda:1')]
        # # tensor([8, 8], device='cuda:0')]
        # print(inputs.shape)
        # # ref: torch.Size([2, 6, 64000])
        # # torch.Size([2, 8, 16000])
        # print(targets.shape)
        # # ref: torch.Size([2, 6, 2, 64000])
        # # torch.Size([2, 8, 4, 16000])
        # print(valid_channels)
        # # ref: tensor([2, 2], device='cuda:0')
        # # ref: tensor([5, 6], device='cuda:0')
        # # ref: tensor([3, 6], device='cuda:0')
        # # ref: tensor([3, 5], device='cuda:0')
        # # tensor([8, 8], device='cuda:0')
        # print(valid_channels.shape)
        # ref: torch.Size([2])
        # torch.Size([2])

        # valid_channels contains a list of valid microphone channels for each example.
        # each example can have a varying number of microphone channels (can come from different arrays).
        # e.g. [[2], [4], [1]] three examples with 2 mics 4 mics and 1 mics.

        # log the metrics
        # loss_, reordered_sources = self.loss_func(est_targets, targets[:, 0], return_est=True)
        # # ref:
        # # mix shape
        # # torch.Size([18, 64000])
        # # mix none shape
        # # torch.Size([1, 18, 64000])
        # # valid_mics shape
        # # torch.Size([1])
        # # valid_mics none shape
        # # torch.Size([1, 1])
        # # print(list(inputs.shape)[0])
        # if list(inputs.shape)[0] == 1:  # If batch size is 1
        #     mix_np = inputs.squeeze(0).cpu().data.numpy()
        #     sources_np = targets[:, 0].squeeze(0).cpu().data.numpy()
        #     est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # else:
        #     mix_np = inputs[:1, :, :].squeeze(0).cpu().data.numpy()
        #     sources_np = targets[:1, :1, :, :].squeeze(0).squeeze(0).cpu().data.numpy()
        #     est_sources_np = reordered_sources[:1, :, :].squeeze(0).cpu().data.numpy()
        # # ref:
        # # mix_np shape
        # # print(mix_np.shape)
        # # (18, 64000)
        # # sources_np shape
        # # print(sources_np.shape)
        # # (4, 64000)
        # # est_sources_np shape
        # # print(est_sources_np.shape)
        # # (4, 64000)

        # utt_metrics = get_metrics(
        #     mix_np[0],
        #     sources_np,
        #     est_sources_np,
        #     sample_rate=16000,
        #     metrics_list=compute_metrics,
        # )
        # return loss, utt_metrics
        return loss


def main(conf):

    train_set = AMIxDataset(
        conf["data"]["train_json"],
        conf["data"]["segment"],
        # num_spk=conf["net"]["n_src"],
        max_mics=conf["main_args"]["max_mics"],
        train=True,
    )
    val_set = AMIxDataset(
        conf["data"]["dev_json"],
        conf["data"]["segment"],
        #   num_spk=conf["net"]["n_src"],
        max_mics=conf["main_args"]["max_mics"],
        train=False,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    try:
        # print(conf["main_args"])
        model_path = os.path.join(conf["main_args"]["exp_dir"], "best_model.pth")
        # print(model_path)
        # load pretrained model
        model = ConvTasNet_mvdr.from_pretrained(model_path)
        print("Load pretrained model")

    except:
        model = ConvTasNet_mvdr(
            **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
        )
        print("Training from scratch")
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=conf["training"]["patience"]
        )
    else:
        scheduler = None
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr)
    system = TACSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=conf["training"]["save_top_k"],
        verbose=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", patience=conf["training"]["patience"], verbose=True
            )
        )

    # Don't ask GPU if they are not available.

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        # plugins=DDPStrategy(find_unused_parameters=False),
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        gradient_clip_val=conf["training"]["gradient_clipping"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    # to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("./local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
