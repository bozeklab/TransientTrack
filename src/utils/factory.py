import os

import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from dataset.dataloader import CocoVidDataLoader
from models.defdetr import DeformableDetr


class Config:
    def __init__(self, config_name, model_name=None):
        with open(os.path.join("src/configs", config_name), "r") as f:
            config = yaml.safe_load(f)

        config["folder"]["output_dir"] = os.path.join(
            config["folder"]["output_dir"], model_name
        )
        config["folder"]["output_training"] = os.path.join(
            config["folder"]["output_dir"], "training"
        )

        config["misc"]["channel"] = (
            config["misc"]["channel"] if "gray" in config["misc"]["img_type"] else None
        )

        self._dict_to_attr(config)

    def _dict_to_attr(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries to Config objects
                setattr(self, key, Config.from_dict(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, dictionary):
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._dict_to_attr(dictionary)
        return instance

    def __repr__(self):
        return f"{self.__dict__}"

    def to_dict(self):
        """Recursively convert Config object to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def save_config(self):
        """Save Config object to a yaml file"""
        with open(os.path.join(self.folder.output_dir, "config.yaml"), "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def create_model(config, train):
    num_channels = 1 if "gray" in config.misc.img_type else 3

    model = DeformableDetr(
        train=train,
        batch_size=config.training.batch_size,
        num_channels=num_channels,
        positional_encoding=config.model.positional_encoding,
        lr=config.training.learning_rate,
        two_stage=config.model.two_stage,
        box_refine=config.model.box_refine,
        bbox_loss_coefficient=config.loss.bbox_loss_coefficient,
        bbox_cost=config.model.bbox_cost,
        classification_loss_coefficient=config.loss.classification_loss_coefficient,
        class_cost=config.model.class_cost,
        giou_loss_coefficient=config.loss.giou_loss_coefficient,
        giou_cost=config.model.giou_cost,
        focal_alpha=config.loss.focal_alpha,
        use_auxiliary_loss=config.loss.use_auxiliary_loss,
        auxiliary_loss_coefficient=config.loss.auxiliary_loss_coefficient,
    )
    return model


def create_dataloader(config, batch_size=None):
    batch_size = batch_size or config.training.batch_size

    dataloader = CocoVidDataLoader(
        data_dir=config.folder.data_dir,
        ann_file=config.folder.ann_file,
        img_type=config.misc.img_type,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seq_length=config.training.seq_length,
        augment=config.training.augment,
        channel=config.misc.channel,
        resize_ratio=config.training.resize_ratio,
    )
    return dataloader


def create_dataset():
    pass


def create_trainer(config, callbacks, logger=None):
    trainer = Trainer(
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator="gpu",
        num_nodes=config.training.num_gpus,
        callbacks=callbacks,
        logger=logger,
        deterministic=config.training.deterministic,
        benchmark=False,
        enable_model_summary=True,
    )

    return trainer


def create_callbacks(config):
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.folder.output_training, "checkpoints"),
        filename="epoch-{epoch}_val-{Val/Loss_epoch:.3f}",
        monitor="Val/Loss_epoch",
        auto_insert_metric_name=False,
        mode="min",
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    if config.training.early_stop_patience:
        early_stop_callback = EarlyStopping(
            monitor="Val/Loss_epoch", patience=config.training.early_stop_patience
        )
        callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]
    else:
        callbacks = [checkpoint_callback, lr_monitor]

    return callbacks
