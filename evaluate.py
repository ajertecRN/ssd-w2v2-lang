import os
import json
import argparse
from pathlib import Path
from loguru import logger
from pprint import pformat

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger

from src.data import get_dataset
from src.augmentation import get_augmenter
from src.models import get_processor, get_model
from src.lightning_modules import (
    get_lightning_module_class,
)
from src.spectrograms import get_spectrogram_fn
from src.callbacks import PredictionsWriter

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_SEQ_LENGTH = 32000  # 1 sec of audio
DEFAULT_NUM_WORKERS = 8
DEFAULT_MULTIPLY_FACTOR = 1
DEFAULT_LENGTH_LIMIT = None

DEFAULT_SPECTROGRAM_KWARGS = {
    "type": "stft",
    "n_fft": 1024,
    "spectrogram_height": 512,
    "spectrogram_width": 512,
}
DEFAULT_SEED = 45


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
        "--model_name",
        type=str,
        required=True,
        help="Model name from HF's hub.",
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
        "--length_limit",
        type=int,
        default=DEFAULT_LENGTH_LIMIT,
        help="Limit number of samples in the dataset",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--use_augmentations",
        action="store_true",
        help="Whether to use augmentations.",
    )
    parser.add_argument(
        "--do_predict",
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
        help="Multiply factor controling number of random windows for single audio sample.",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["hft", "timm"],
        required=True,
        help="Model source, hft (huggingface transformers) or timm (pytorch-image-models).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to a model.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        required=True,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=False,
        help="Output dir.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="",
        required=False,
        help="Output dir for predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="eval seed -- for reproducibility.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.accelerator == "gpu" else ""

    seed_everything(45)

    logger.info(f"Arguments:\n{pformat(vars(args), indent=2)}")
    output_folder = (
        Path(args.output_dir) if args.output_dir else Path(args.model_path).parent
    )

    logger.info(f"Setting seed to: {args.seed}")
    seed_everything(args.seed, workers=True)

    num_labels = len(args.possible_labels)

    config, processor, model = get_model(
        model_source=args.model_source,
        model_name_or_path=args.model_name,
        num_labels=num_labels,
        sample_rate=args.sample_rate,
        freeze_feature_encoder=False,
    )

    lightning_module = get_lightning_module_class(model_source=args.model_source)

    model_lightning = lightning_module.load_from_checkpoint(
        checkpoint_path=args.model_path,
        model=model,
        input_args=args,
        num_classes=num_labels,
    )
    model_lightning.eval()

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

    evaluation_dataset = get_dataset(
        filepath=args.eval_filepath,
        processor=processor,
        augmenter=augmenter,
        spectrogram_fn=spectrogram_fn,
        args=args,
    )

    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    if args.do_predict:
        if args.predictions_dir:
            Path(args.predictions_dir).mkdir(parents=True, exist_ok=True)
            output_folder = args.predictions_dir

        trainer = Trainer(
            accelerator=args.accelerator,
            auto_select_gpus=False,
            devices=1,  # NOTE: recommended
            logger=CSVLogger(save_dir=output_folder, name="prediction_logs"),
            callbacks=PredictionsWriter(
                output_folder,
                write_interval="epoch",
            ),
        )
        trainer.predict(model=model_lightning, dataloaders=evaluation_dataloader)

    else:
        trainer = Trainer(
            accelerator=args.accelerator,
            auto_select_gpus=False,
            devices=1,  # NOTE: recommended
            logger=CSVLogger(save_dir=output_folder, name="evaluation_logs"),
        )
        trainer.test(
            model=model_lightning, dataloaders=evaluation_dataloader, verbose=True
        )


if __name__ == "__main__":
    main()
