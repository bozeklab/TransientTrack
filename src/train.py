import argparse
import os
from datetime import datetime

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

import utils.factory as Factory
from utils.factory import Config


def train(cfg, name, log, project_name=""):
    model = Factory.create_model(cfg, train=True)
    dataloader = Factory.create_dataloader(cfg)

    os.makedirs(cfg.folder.output_training, exist_ok=True)

    # logging
    if log:
        wandb_logger = WandbLogger(
            project=project_name,
            name=name + "_" + str(datetime.now()),
        )

        tmp_conf = cfg.to_dict()
        for key in tmp_conf.keys():
            wandb_logger.log_hyperparams(tmp_conf[key])
    else:
        wandb_logger = None

    if cfg.training.save_checkpoints:
        os.makedirs(
            os.path.join(cfg.folder.output_training, "checkpoints"), exist_ok=True
        )

    # add mean and std to config
    cfg.training.mean = dataloader.mean
    cfg.training.std = dataloader.std

    # save config
    cfg.save_config()

    callbacks = Factory.create_callbacks(cfg)

    trainer = Factory.create_trainer(cfg, callbacks, wandb_logger)

    print("Starting training at {}".format(datetime.now()))

    # train
    trainer.fit(model, datamodule=dataloader)

    print("Training finished at {}".format(datetime.now()))

    trainer.test(model, datamodule=dataloader)

    # final tests for tracking
    print("Starting tracking evaluation")
    print(
        f"Using checkpoint {trainer.checkpoint_callback.best_model_path} for tracking evaluation"
    )


if __name__ == "__main__":
    SEED = 42
    seed_everything(SEED, True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--project-name", default="", type=str)

    args = parser.parse_args()

    if not args.no_log and args.project_name == "":
        raise ValueError(
            "If logging is enabled, a project name must be provided with --project-name"
        )

    cfg = Config("config.yaml", args.name)

    train(cfg, args.name, not args.no_log, project_name=args.project_name)
