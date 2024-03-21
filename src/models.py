from loguru import logger

import torch
import timm
from torchvision import transforms

from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
)


def get_hf_model(
    model_name_or_path: str,
    num_labels: int,
    sample_rate: int,
    freeze_feature_encoder: bool = False,
):
    logger.info(f"Loading model: {model_name_or_path}")
    config = Wav2Vec2Config.from_pretrained(model_name_or_path, num_labels=num_labels)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path, sampling_rate=sample_rate
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name_or_path, config=config
    )

    if freeze_feature_encoder:
        logger.info("Freezing model's feature encoder.")
        model.freeze_feature_encoder()

    return config, processor, model


def get_processor(processor_name_or_path: str, sample_rate: int):
    return Wav2Vec2FeatureExtractor.from_pretrained(
        processor_name_or_path, sampling_rate=sample_rate
    )


def get_timm_model(
    model_name: str, num_classes: int, in_channels: int = 1, pretrained: bool = True
):
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_channels
    )
    config = model.default_cfg

    processor = transforms.Compose([transforms.ToTensor()])

    return config, processor, model


def get_model(
    model_source: str,
    model_name_or_path: str,
    num_labels: int,
    sample_rate: int = 16000,
    freeze_feature_encoder: bool = False,
    pretrained: bool = False,
    in_channels: int = 1,
):
    if model_source == "hft":
        return get_hf_model(
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
            sample_rate=sample_rate,
            freeze_feature_encoder=freeze_feature_encoder,
        )
    elif model_source == "timm":
        return get_timm_model(
            model_name=model_name_or_path,
            num_classes=num_labels,
            in_channels=in_channels,
            pretrained=pretrained,
        )
    else:
        raise ValueError
