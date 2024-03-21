import os
import argparse
from pathlib import Path
from loguru import logger
from pprint import pformat
import json

import torch
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from src.data import get_dataset
from src.augmentation import get_augmenter
from src.models import get_model
from src.lightning_modules import get_lightning_module
from src.spectrograms import get_spectrogram_fn
from src.utils import save_args

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_SEQ_LENGTH = 32000  # 2 sec of audio
DEFAULT_NUM_WORKERS = 8
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_GRADIENT_ACC_STEPS = 1
DEFAULT_SEED = 45
DEFAULT_CALLBACK_MONITOR = "f1_macro_dev"
DEFAULT_CALLBACK_MODE = "max"
DEFAULT_EARLY_STOP_DELTA = 1e-4
DEFAULT_EARLY_STOP_PATIENCE = 3
DEFAULT_MULTIPLY_FACTOR = 3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_LENGTH_LIMIT = None
DEFAULT_LOG_STEPS = 50
DEFAULT_SPECTROGRAM_KWARGS = {
    "type": "stft",
    "n_fft": 1024,
    "spectrogram_height": 512,
    "spectrogram_width": 512,
}
DEFAULT_SAVE_TOP_K = 3
DEFAULT_LR_SCHEDULER_KWARGS = {
    "type": "reduce_on_plateau",
    "factor": 0.1,
    "patience": 2,
    "cooldown": 1,
    "mode": DEFAULT_CALLBACK_MODE,
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--accelerator", required=True, choices=["cpu", "gpu"], type=str, help="."
    )
    parser.add_argument("--gpu", required=True, type=str, help=".")
    parser.add_argument(
        "--devices", required=True, type=int, help="Number of devices to use."
    )

    parser.add_argument(
        "--train_filepath",
        type=str,
        required=True,
        help="Path to training dataset.",
    )
    parser.add_argument(
        "--eval_filepath",
        type=str,
        required=True,
        help="Path to evaluation dataset.",
    )

    parser.add_argument(
        "--audio_filepart_column",
        type=str,
        required=True,
        help="Column with relative path to audio files",
    )
    parser.add_argument(
        "--labels_column",
        type=str,
        required=True,
        help="Column with labels.",
    )
    parser.add_argument(
        "--audio_root_dir",
        type=str,
        required=True,
        help="Path to directory containing audio files.",
    )
    parser.add_argument(
        "--drop_duplicates",
        action="store_true",
        help="Whether to drop data duplicates.",
    )

    # TODO:
    parser.add_argument(
        "--possible_labels",
        nargs="+",
        required=True,
        help="",
    )

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum audio sequence length; max number of audio samples.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--length_limit",
        type=int,
        default=DEFAULT_LENGTH_LIMIT,
        help="Limit number of samples in the dataset",
    )
    parser.add_argument(
        "--use_augmentations",
        action="store_true",
        help="Whether to use augmentations.",
    )
    parser.add_argument(
        "--augmentations_config_path",
        type=str,
        default=None,
        help="Path to yaml file containing augmentations configuration.",
    )
    parser.add_argument(
        "--use_spectrograms",
        action="store_true",
        help="Whether to use spectrograms as input values into models.",
    )
    parser.add_argument(
        "--spectrogram_kwargs",
        type=json.loads,
        default=DEFAULT_SPECTROGRAM_KWARGS,
        help="Spectrogram kwargs.",
    )

    parser.add_argument(
        "--multiply_sample_and_cache_random_window",
        action="store_true",
        help="Whether to sample multiple random windows (controled by `--multiply_factor` flag) from single audio sample, and chache them for subsequent epochs.",
    )
    parser.add_argument(
        "--multiply_factor",
        type=int,
        default=DEFAULT_MULTIPLY_FACTOR,
        help="Multiply factor controling number of random windows fsor single audio sample.",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["hft", "timm"],
        required=True,
        help="Model source, hft (huggingface transformers) or timm (pytorch-image-models).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name from HF's hub or local path to a model.",
    )
    parser.add_argument(
        "--freeze_feature_encoder",
        action="store_true",
        help="Whether to freeze models feature encoder.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        required=True,
        help="Training batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        required=True,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Training learning rate.",
    )
    parser.add_argument(
        "--scheduler_kwargs",
        type=json.loads,
        default=DEFAULT_LR_SCHEDULER_KWARGS,
        required=False,
        help="LR scheduler kwargs.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output dir for model's artifacts (ckpt, training params, etc.).",
    )
    parser.add_argument(
        "--logs_folder",
        type=str,
        default=None,
        required=False,
        help="Path to logs dir.",
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=DEFAULT_SAVE_TOP_K,
        required=False,
        help="Number of top ckpts to save. -1 for saving all, 0 for saving none.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help="Max gradient norm. Everything above will be clipped.",
    )

    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=DEFAULT_LOG_STEPS,
        help="How often to log, every N number of steps.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_GRADIENT_ACC_STEPS,
        help="Gradient accumulation steps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Training seed -- for reproducibility.",
    )

    parser.add_argument(
        "--callback_monitor",
        type=str,
        default=DEFAULT_CALLBACK_MONITOR,
        help="Callback monitor key (metric).",
    )
    parser.add_argument(
        "--callback_mode",
        type=str,
        choices=["max", "min"],
        default=DEFAULT_CALLBACK_MODE,
        help="Callback mode.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=DEFAULT_EARLY_STOP_DELTA,
        help="Early stopping minimum delta.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Early stopping patience.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.accelerator == "gpu" else ""

    assert not (
        Path(args.output_dir).exists() and Path(args.output_dir).is_dir()
    ), f"Folder '{Path(args.output_dir).resolve()}' exists."

    if args.logs_folder is not None:
        logs_folder = Path(args.logs_folder)
    else:
        logs_folder = Path(args.output_dir)

    logger.add(os.path.join(str(logs_folder), "logger_logs.out"))

    logger.info(f"Training arguments:\n{pformat(vars(args), indent=2)}")
    save_args(args, output_dir=args.output_dir)

    logger.info(f"Setting seed to: {args.seed}")
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True

    num_labels = len(args.possible_labels)
    logger.info(
        f"Number of labels: {num_labels}. Possible labels: {args.possible_labels}"
    )

    config, processor, model = get_model(
        model_source=args.model_source,
        model_name_or_path=args.model_name_or_path,
        num_labels=num_labels,
        sample_rate=args.sample_rate,
        freeze_feature_encoder=args.freeze_feature_encoder,
        pretrained=True,  # timm pretrained
    )

    if args.use_augmentations:
        logger.info(
            f"Using augmentations. Config path: {args.augmentations_config_path}"
        )
        assert Path(args.augmentations_config_path).is_file()
        augmenter = get_augmenter(
            config_path=args.augmentations_config_path, sample_rate=args.sample_rate
        )
        logger.info(f"Augmentation config:\n{pformat(augmenter.config, indent=2)}")
    else:
        augmenter = None

    if args.use_spectrograms:
        assert args.model_source == "timm", "Spectrograms work only with timm models."
        logger.info(
            f"Using spectrograms. Config:\n{pformat(dict(args.spectrogram_kwargs), indent=2)}"
        )
        spectrogram_fn = get_spectrogram_fn(
            args.spectrogram_kwargs["type"], spectrogram_kwargs=args.spectrogram_kwargs
        )
    else:
        spectrogram_fn = None

    training_dataset = get_dataset(
        filepath=args.train_filepath,
        processor=processor,
        augmenter=augmenter,
        spectrogram_fn=spectrogram_fn,
        args=args,
    )

    evaluation_dataset = get_dataset(
        filepath=args.eval_filepath,
        processor=processor,
        augmenter=augmenter,
        spectrogram_fn=spectrogram_fn,
        args=args,
    )

    training_sampler = RandomSampler(training_dataset)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=args.train_batch_size,
        sampler=training_sampler,
        num_workers=args.num_workers,
    )

    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model_lightning = get_lightning_module(
        model_source=args.model_source,
        model=model,
        input_args=args,
        num_classes=num_labels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_monitor=args.callback_monitor,
        scheduler_kwargs=args.scheduler_kwargs,
    )

    csv_logger = CSVLogger(save_dir=logs_folder, name="training_logs")

    trainer = Trainer(
        accelerator=args.accelerator,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        auto_select_gpus=False,
        callbacks=[
            EarlyStopping(
                monitor=args.callback_monitor,
                min_delta=args.early_stopping_min_delta,
                patience=args.early_stopping_patience,
                verbose=True,
                mode=args.callback_mode,
                check_on_train_epoch_end=False,
            ),
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename="epoch_{epoch:03d}-"
                + f"{args.callback_monitor}"
                + "_{"
                + f"{args.callback_monitor}"
                + ":.4f}",
                monitor=args.callback_monitor,
                save_last=True,
                save_top_k=args.save_top_k,
                mode=args.callback_mode,
                auto_insert_metric_name=False,
                save_weights_only=False,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            ),
            TQDMProgressBar(refresh_rate=10),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        devices=args.devices,
        enable_checkpointing=True,
        gradient_clip_val=args.max_grad_norm,
        gradient_clip_algorithm="norm",
        logger=csv_logger,
        max_epochs=args.n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model=model_lightning,
        train_dataloaders=training_dataloader,
        val_dataloaders=evaluation_dataloader,
    )


if __name__ == "__main__":
    main()
