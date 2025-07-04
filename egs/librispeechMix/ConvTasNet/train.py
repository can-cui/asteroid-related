import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import PyTorchProfiler

from asteroid.models import ConvTasNet
from asteroid.data import LibriMix
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
from local.dataset.data_loader_dynamicMix import SpectrogramDataset, BucketingSampler, AudioDataLoader, _collate_fn

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


def main(conf):
    # train_set = LibriMix(
    #     csv_dir=conf["data"]["train_dir"],
    #     task=conf["data"]["task"],
    #     sample_rate=conf["data"]["sample_rate"],
    #     n_src=conf["data"]["n_src"],
    #     segment=conf["data"]["segment"],
    # )

    # val_set = LibriMix(
    #     csv_dir=conf["data"]["valid_dir"],
    #     task=conf["data"]["task"],
    #     sample_rate=conf["data"]["sample_rate"],
    #     n_src=conf["data"]["n_src"],
    #     segment=conf["data"]["segment"],
    # )

    # train_loader = DataLoader(
    #     train_set,
    #     shuffle=True,
    #     batch_size=conf["training"]["batch_size"],
    #     num_workers=conf["training"]["num_workers"],
    #     drop_last=True,
    # )

    # val_loader = DataLoader(
    #     val_set,
    #     shuffle=False,
    #     batch_size=conf["training"]["batch_size"],
    #     num_workers=conf["training"]["num_workers"],
    #     drop_last=True,
    # )
    train_data = SpectrogramDataset(
        conf["data"]["data_path"],
        manifest_filepath_list=conf["data"]["train_dir"],
        segment=conf["data"]["segment"],
        normalize=True,
        augment=False)
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        collate_fn=_collate_fn,
        drop_last=True,
    )
    valid_data = SpectrogramDataset(
        conf["data"]["data_path"],
        manifest_filepath_list=conf["data"]["valid_dir"],
        segment=conf["data"]["segment"],
        normalize=True,
        augment=False)
    val_loader = AudioDataLoader(valid_data,
                                 num_workers=conf["training"]["num_workers"],
                                 batch_size=conf["training"]["batch_size"],
                                 collate_fn=_collate_fn,
                                 drop_last=True)



    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    model = ConvTasNet(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
    )
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))
    # profiler = PyTorchProfiler(dirpath=exp_dir,
    #                            filename="profiler.txt",
    #                            sort_by_key="cuda_memory_usage")

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        # profiler=profiler
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
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
