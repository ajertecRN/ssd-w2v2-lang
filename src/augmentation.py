from typing import Dict, Union, Optional, Callable, List
from pathlib import Path
from loguru import logger

import numpy as np

import audiomentations
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations import (
    Compose,
    SpecCompose,
    AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    AirAbsorption,
    ClippingDistortion,
    Mp3Compression,
    PeakingFilter,
    RoomSimulator,
    SevenBandParametricEQ,
    TanhDistortion,
    TimeMask,
    PitchShift,
    Shift,
    TimeStretch,
    ApplyImpulseResponse,
    AddShortNoises,
    SpecChannelShuffle,
    SpecFrequencyMask,
)

from .augmentation_utils import (
    gen_notch_coeffs,
    filter_FIR,
    normalize_wav,
    random_range,
)
from .utils import load_yaml


class RawBoostLNL(BaseWaveformTransform):
    """
    From paper: RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing
    (https://arxiv.org/abs/2111.04433)
    implementation: https://github.com/TakHemlata/RawBoost-antispoofing

    linear and non-linear convolutive noise
    """

    def __init__(
        self,
        p: float = 0.5,
        N_f: int = 5,
        nBands: int = 5,
        minF: int = 20,
        maxF: int = 8000,
        minBW: int = 100,
        maxBW: int = 1000,
        minCoeff: int = 10,
        maxCoeff: int = 100,
        minG: int = 0,
        maxG: int = 0,
        minBiasLinNonLin: int = 5,
        maxBiasLinNonLin: int = 20,
    ):
        """
        Params:
        -------

        N_f: int (default 5)
            - order of the (non-)linearity where N_f=1 refers only to linear components.
        nBands: int (default 5)
            - number of notch filters. The higher the number of bands, the more aggresive the distortions is.
        minF: int (default 20)
            - minimum frequency
        maxF: int (default 8000)
            - maximum frequency
        minBW: int (default 100)
            - min width [Hz] of filter
        maxBW: int (defualt 1000)
            - maximum width [Hz] of filter
        minCoeff: int (default 10)
            - minimum filter coefficients. More the filter coefficients more ideal the filter slope
        maxCoeff: int (default 100)
            - max filter coefficients
        minG: int (default 0)
            - minimum gain factor of linear component
        maxG: int (default 0)
            - max gain factor of linear component.
        minBiasLinNonLin: int (default 5)
            - minimum gain difference between linear and non-linear components
        maxBiasLinNonLin: int (default 20)
            - maximum gain difference between linear and non-linear components

        """

        super().__init__(p)

        self.N_f = N_f
        self.nBands = nBands
        self.minF = minF
        self.maxF = maxF
        self.minBW = minBW
        self.maxBW = maxBW
        self.minCoeff = minCoeff
        self.maxCoeff = maxCoeff
        self.minG = minG
        self.maxG = maxG
        self.minBiasLinNonLin = minBiasLinNonLin
        self.maxBiasLinNonLin = maxBiasLinNonLin

    def apply(self, samples: np.ndarray, sample_rate: int):
        return RawBoostLNL.lnl_convolutive_noise(
            samples,
            fs=sample_rate,
            N_f=self.N_f,
            nBands=self.nBands,
            minF=self.minF,
            maxF=self.maxF,
            minBW=self.minBW,
            maxBW=self.maxBW,
            minCoeff=self.minCoeff,
            maxCoeff=self.maxCoeff,
            minG=self.minG,
            maxG=self.maxG,
            minBiasLinNonLin=self.minBiasLinNonLin,
            maxBiasLinNonLin=self.maxBiasLinNonLin,
        )

    @staticmethod
    def lnl_convolutive_noise(
        x,
        fs,
        N_f,
        nBands,
        minF,
        maxF,
        minBW,
        maxBW,
        minCoeff,
        maxCoeff,
        minG,
        maxG,
        minBiasLinNonLin,
        maxBiasLinNonLin,
    ):
        y = [0] * x.shape[0]
        for i in range(0, N_f):
            if i == 1:
                minG = minG - minBiasLinNonLin
                maxG = maxG - maxBiasLinNonLin
            b = gen_notch_coeffs(
                nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
            )
            y = y + filter_FIR(np.power(x, (i + 1)), b)
        y = y - np.mean(y)
        y = normalize_wav(y, False)
        return y.astype(np.float32)


class RawBoostISD(BaseWaveformTransform):
    """
    From paper: RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing
    (https://arxiv.org/abs/2111.04433)
    implementation: https://github.com/TakHemlata/RawBoost-antispoofing

    impulsive signal dependent noise
    """

    def __init__(self, p: float = 0.5, perc_uniform_samples: int = 10, g_sd: int = 2):
        """
        Params:
        -------

        perc_uniform_samples: int (default 10)
            - Maximum number of uniformly distributed samples in [%]
        g_sd: int (default 2)
            - noise gain parameter > 0
        """

        super().__init__(p)

        self.perc_uniform_samples = perc_uniform_samples
        self.g_sd = g_sd

    def apply(self, samples: np.ndarray, sample_rate: int):
        return RawBoostISD.isd_additive_noise(
            samples,
            P=self.perc_uniform_samples,
            g_sd=self.g_sd,
        )

    @staticmethod
    def isd_additive_noise(x, P, g_sd):
        beta = random_range(0, P, False)

        y = x.copy()
        x_len = x.shape[0]
        n = int(x_len * (beta / 100))
        p = np.random.permutation(x_len)[:n]
        f_r = np.multiply(
            ((2 * np.random.rand(p.shape[0])) - 1),
            ((2 * np.random.rand(p.shape[0])) - 1),
        )
        r = g_sd * x[p] * f_r
        y[p] = x[p] + r
        y = normalize_wav(y, False)
        return y.astype(np.float32)


class RawBoostSSI(BaseWaveformTransform):
    """
    From paper: RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing
    (https://arxiv.org/abs/2111.04433)
    implementation: https://github.com/TakHemlata/RawBoost-antispoofing

    stationary signal independent noise
    """

    def __init__(
        self,
        p: float = 0.5,
        nBands: int = 5,
        minF: int = 20,
        maxF: int = 8000,
        minBW: int = 100,
        maxBW: int = 1000,
        minCoeff: int = 10,
        maxCoeff: int = 100,
        minG: int = 0,
        maxG: int = 0,
        SNRmin: int = 10,
        SNRmax: int = 40,
    ):
        """
        Params:
        -------

        nBands: int (default 5)
            - number of notch filters. The higher the number of bands, the more aggresive the distortions is.
        minF: int (default 20)
            - minimum frequency
        maxF: int (default 8000)
            - maximum frequency
        minBW: int (default 100)
            - min width [Hz] of filter
        maxBW: int (defualt 1000)
            - maximum width [Hz] of filter
        minCoeff: int (default 10)
            - minimum filter coefficients. More the filter coefficients more ideal the filter slope
        maxCoeff: int (default 100)
            - max filter coefficients
        minG: int (default 0)
            - minimum gain factor of linear component
        maxG: int (default 0)
            - max gain factor of linear component.
        SNRmin: int (default 10)
            - minimum gain difference between linear and non-linear components
        SNRmax: int (default 40)
            - maximum gain difference between linear and non-linear components

        """

        super().__init__(p)

        self.nBands = nBands
        self.minF = minF
        self.maxF = maxF
        self.minBW = minBW
        self.maxBW = maxBW
        self.minCoeff = minCoeff
        self.maxCoeff = maxCoeff
        self.minG = minG
        self.maxG = maxG
        self.SNRmin = SNRmin
        self.SNRmax = SNRmax

    def apply(self, samples: np.ndarray, sample_rate: int):
        return RawBoostSSI.ssi_additive_noise(
            samples,
            fs=sample_rate,
            nBands=self.nBands,
            minF=self.minF,
            maxF=self.maxF,
            minBW=self.minBW,
            maxBW=self.maxBW,
            minCoeff=self.minCoeff,
            maxCoeff=self.maxCoeff,
            minG=self.minG,
            maxG=self.maxG,
            SNRmin=self.SNRmin,
            SNRmax=self.SNRmax,
        )

    @staticmethod
    def ssi_additive_noise(
        x,
        fs,
        nBands,
        minF,
        maxF,
        minBW,
        maxBW,
        minCoeff,
        maxCoeff,
        minG,
        maxG,
        SNRmin,
        SNRmax,
    ):
        noise = np.random.normal(0, 1, x.shape[0])
        b = gen_notch_coeffs(
            nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
        )
        noise = filter_FIR(noise, b)
        noise = normalize_wav(noise, True)
        SNR = random_range(SNRmin, SNRmax, False)
        noise = (
            noise
            / np.linalg.norm(noise, 2)
            * np.linalg.norm(x, 2)
            / 10.0 ** (0.05 * SNR)
        )
        x = x + noise
        return x.astype(np.float32)


class AddBackgroundNoiseCached(AddBackgroundNoise):
    def __init__(
        self,
        sounds_path: Union[List[Path], List[str], Path, str],
        sample_rate: int,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        noise_rms: str = "relative",
        min_absolute_rms_in_db: float = -45.0,
        max_absolute_rms_in_db: float = -15.0,
        noise_transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        p: float = 0.5,
        lru_cache_size: int = 2,
    ):
        super().__init__(
            sounds_path,
            min_snr_in_db,
            max_snr_in_db,
            noise_rms,
            min_absolute_rms_in_db,
            max_absolute_rms_in_db,
            noise_transform,
            p,
            lru_cache_size,
        )

        self.sample_rate = sample_rate

        self.audio_cache: Dict[str, np.ndarray] = dict()
        for path in self.sound_file_paths:
            logger.info(f"Loading background audio file: '{path}'")

            audio, sr = audiomentations.core.audio_loading_utils.load_sound_file(
                path, sample_rate=self.sample_rate
            )
            self.audio_cache[path] = (audio.astype(np.float32), sr)

        self._load_sound = lambda file_path, sample_rate: self.audio_cache[file_path]


AUGMENTATION_NAME_TO_CLASS = {
    # time related:
    "background_noise": AddBackgroundNoiseCached,
    "impulse_response": ApplyImpulseResponse,
    "short_noises": AddShortNoises,
    "gaussian_noise": AddGaussianNoise,
    "gaussian_snr": AddGaussianSNR,
    "air_absorption": AirAbsorption,
    "clipping_distortion": ClippingDistortion,
    "mp3_compression": Mp3Compression,
    "biquad_peaking_filter": PeakingFilter,
    "room_simulator": RoomSimulator,
    "seven_band_parametric_eq": SevenBandParametricEQ,
    "tanh_distortion": TanhDistortion,
    "time_mask": TimeMask,
    "pitch_shift": PitchShift,
    "shift": Shift,
    "time_stretch": TimeStretch,
    # on spectrograms:
    "spectrogram_channel_shuffle": SpecChannelShuffle,
    "spectrogram_frequency_mask": SpecFrequencyMask,
    # rawboost:
    "rawboost_lnl": RawBoostLNL,
    "rawboost_isd": RawBoostISD,
    "rawboost_ssi": RawBoostSSI,
}


class AudioAugmenter:
    def __init__(
        self,
        config_path_or_dict: Union[str, dict],
        sample_rate: int,
    ):
        self.config_path_or_dict = config_path_or_dict
        self.sample_rate = sample_rate

        if isinstance(config_path_or_dict, str):
            self.config = load_yaml(config_path_or_dict)
        elif isinstance(config_path_or_dict, dict):
            self.config = config_path_or_dict
        else:
            raise ValueError

        time_augment_list = self.parse_config(
            config=self.config, key="time_augmentations"
        )
        self.time_augment = Compose(time_augment_list)

        spec_augment_list = self.parse_config(
            config=self.config, key="spectrogram_augmentations"
        )
        self.spec_augment = SpecCompose(spec_augment_list)

    def parse_config(
        self, config, key: str, map_name_class: dict = AUGMENTATION_NAME_TO_CLASS
    ):
        if key in config:
            return [
                map_name_class[name](**params) for name, params in config[key].items()
            ]
        else:
            return []

    def time_transform(self, audio: np.ndarray, **kwargs):
        return self.time_augment(samples=audio, sample_rate=self.sample_rate)

    def spectral_transform(self, spectrum: np.ndarray, *args, **kwargs):
        return self.spec_augment(magnitude_spectrogram=spectrum)


def get_augmenter(
    config_path: str,
    sample_rate: int,
):
    return AudioAugmenter(config_path_or_dict=config_path, sample_rate=sample_rate)
