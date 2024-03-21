from typing import Callable
from loguru import logger

import numpy as np
import librosa
import skimage

from .utils import minmax_scale, img_y_log_scale


def transform_stft(**kwargs):
    n_fft = kwargs.get("n_fft", 1024)
    spectrogram_height = kwargs.get("spectrogram_height", 512)
    spectrogram_width = kwargs.get("spectrogram_width", 512)

    logger.info(
        f"Using STFT: {dict(n_fft=n_fft, spectrogram_height=spectrogram_height, spectrogram_width=spectrogram_width)}"
    )

    def transform(arr: np.ndarray):
        if arr.any():
            stft = np.abs(librosa.stft(arr, n_fft=n_fft))
            stft = librosa.amplitude_to_db(stft, ref=np.max)
            stft = img_y_log_scale(stft, spectrogram_width, spectrogram_height, base=2)

            # stft = minmax_scale(stft, 0, 255, np.float32)
            stft = (stft - stft.mean()) / stft.std()
        else:
            stft = np.zeros((spectrogram_width, spectrogram_height))

        return stft.astype(np.float32)

    return transform


def transform_cqt(**kwargs):
    sample_rate = kwargs.get("sample_rate", 16000)
    hop_length = kwargs.get("hop_length", 512)
    n_bins = kwargs.get("n_bins", 152)
    bins_per_octave = kwargs.get("bins_per_octave", 22)
    spectrogram_height = kwargs.get("spectrogram_height", 512)
    spectrogram_width = kwargs.get("spectrogram_width", 512)

    logger.info(
        f"Using CQT: {dict(sample_rate=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, spectrogram_height=spectrogram_height, spectrogram_width=spectrogram_width)}"
    )

    def transform(arr: np.ndarray):
        if arr.any():
            cqt = np.abs(
                librosa.cqt(
                    arr,
                    sr=sample_rate,
                    hop_length=hop_length,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave,
                )
            )
            cqt = librosa.amplitude_to_db(cqt, ref=np.max)
            cqt = skimage.transform.resize(cqt, (spectrogram_height, spectrogram_width))

            # cqt = minmax_scale(cqt, 0, 255, np.float32)
            cqt = (cqt - cqt.mean()) / cqt.std()
        else:
            cqt = np.zeros((spectrogram_height, spectrogram_width))

        return cqt.astype(np.float32)

    return transform


def get_spectrogram_fn(spectrogram_type: str, spectrogram_kwargs: dict) -> Callable:
    if spectrogram_type == "stft":
        return transform_stft(**spectrogram_kwargs)
    elif spectrogram_type == "cqt":
        return transform_cqt(**spectrogram_kwargs)
    else:
        raise ValueError
