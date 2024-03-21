import os
from typing import Union, Optional, List
import pathlib
from pathlib import Path
from loguru import logger
from pprint import pformat

import librosa
import pandas as pd
import pyarrow as pa
import tqdm
import numpy as np
import soundfile as sf

import torch
from torch.utils.data import Dataset


def read_data(
    filepath: Union[str, pathlib.PosixPath],
    usecols: Optional[List[str]] = None,
    length_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Reads a structured '.csv' or '.tsv' file. Uses pandas."""

    if str(filepath).endswith(".csv"):
        sep = ","
    elif str(filepath).endswith(".tsv"):
        sep = "\t"
    else:
        raise ValueError("Wrong file format. Expected .csv or .tsv.")

    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(
        filepath,
        sep=sep,
        nrows=length_limit if length_limit is not None else None,
        usecols=usecols if usecols is not None else None,
    )

    logger.info("Input dataset (head):")
    logger.info(f"\n{df.head()}")
    return df


class AudioMultiClassDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        audio_filepart_column: str,
        labels_column: str,
        audio_root_dir: str,
        processor,
        possible_labels: list,
        max_sequence_length: int,
        sample_rate: int,
        padding: Union[bool, str] = "max_length",
        truncation: Union[bool, str] = True,
        length_limit: Optional[int] = None,
        drop_duplicates: bool = False,
        use_augmentations: bool = False,
        augmenter=None,
        multiply_sample_and_cache_random_window: bool = False,
        multiply_factor: int = 1,
        use_spectrograms: bool = False,
        spectrogram_fn=None,
        **kwargs,
    ):
        """

        Parameters:
        -----------

        max_sequence_length: int
            - max length of audio, in number of samples
            NOTE: (if sample rate is 16kHz, truncation=True, and audio length is 3 sec (3*16k samples), max_length of 16000 will limit audio to only 1 sec; similar for padding)

        audio_filepart_column: str
            - csv column containing filenames (e.g. '1234.mp3') or filepath parts (e.g. 'libri_16khz/123.mp3') to the audio files inside the `audio_root_dir` .


        `audio_root_dir` and `audio_filepart_column` form full path to audio files.

        """
        super().__init__()

        self.filepath = filepath
        self.audio_filepart_column = audio_filepart_column
        self.labels_column = labels_column
        self.audio_root_dir = audio_root_dir
        self.processor = processor
        self.possible_labels = possible_labels
        self.max_sequence_length = max_sequence_length
        self.sample_rate = sample_rate
        self.padding = padding
        self.truncation = truncation
        self.length_limit = length_limit
        self.drop_duplicates = drop_duplicates
        self.use_augmentations = use_augmentations
        self.augmenter = augmenter
        self.multiply_sample_and_cache_random_window = (
            multiply_sample_and_cache_random_window
        )
        self.multiply_factor = multiply_factor
        self.use_spectrograms = use_spectrograms
        self.spectrogram_fn = spectrogram_fn

        self.num_labels = len(self.possible_labels)

        self.labels_to_idx_map = {}
        for idx, label_name in enumerate(self.possible_labels):
            self.labels_to_idx_map[str(label_name)] = idx

        logger.info(f"Label to index map: {pformat(self.labels_to_idx_map, indent=4)}")

        self.audio_filepaths, self.labels = self.get_audio_paths_and_labels()
        assert len(self.audio_filepaths) == len(self.labels)

        self.random_start_indices = None
        if self.multiply_sample_and_cache_random_window:
            logger.info(
                f"Sampling all audios '{self.multiply_factor}x' times with random windows and caching these windows."
            )

            self.random_start_indices = []
            for audio_filepath in tqdm.tqdm(
                self.audio_filepaths, desc="Loading audio & computing random windows"
            ):
                # audio, sr = librosa.load(str(audio_filepath), sr=self.sample_rate)
                audio, sr = sf.read(str(audio_filepath))
                len_audio = len(audio)

                for _ in range(int(self.multiply_factor)):
                    if len_audio <= self.max_sequence_length:
                        random_start_idx = 0
                    else:
                        end_bound_idx = max(
                            len_audio - self.max_sequence_length,
                            self.max_sequence_length,
                        )
                        random_start_idx = np.random.randint(0, end_bound_idx)

                    self.random_start_indices.append(random_start_idx)

            self.random_start_indices = np.array(self.random_start_indices)

            self.audio_filepaths = np.repeat(
                self.audio_filepaths.to_numpy(zero_copy_only=False),
                self.multiply_factor,
            )

            self.labels = np.repeat(self.labels, self.multiply_factor)

            assert len(self.random_start_indices) == len(self.audio_filepaths)
        else:
            logger.info(
                f"Using first '{self.max_sequence_length}' samples from each audio file."
            )

    def get_audio_paths_and_labels(self):
        df = read_data(
            self.filepath,
            usecols=[self.audio_filepart_column, self.labels_column],
            length_limit=self.length_limit,
        )

        len_df_before_dropna = len(df)
        df.dropna(subset=[self.audio_filepart_column, self.labels_column], inplace=True)
        len_df_after_dropna = len(df)

        logger.info(
            f"Number of dropped NaN rows: {len_df_before_dropna-len_df_after_dropna}"
        )

        if self.drop_duplicates:
            len_df_before_drop_duplicates = len(df)
            df.drop_duplicates(subset=self.audio_filepart_column, inplace=True)
            len_df_after_drop_duplicates = len(df)

            logger.info(
                f"Number of duplicate rows dropped: {len_df_before_drop_duplicates-len_df_after_drop_duplicates}"
            )

        df.reset_index(drop=True, inplace=True)

        logger.info(f"Number of rows in the dataset: {len(df)}")

        return (
            pa.array(
                [
                    os.path.join(self.audio_root_dir, filepart)
                    for filepart in df[self.audio_filepart_column]
                ]
            ),
            df[self.labels_column]
            .astype(str)
            .apply(lambda x: self.labels_to_idx_map[x])
            .values.astype(int),
        )

    def process_audio(self, processor, audio, label, filepath):
        encoded = processor(
            raw_speech=audio,
            max_length=self.max_sequence_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )

        encoded["input_values"] = encoded["input_values"].squeeze()
        encoded["labels"] = torch.tensor(label)

        if "attention_mask" in encoded:
            encoded["attention_mask"] = encoded["attention_mask"].squeeze()

        out = dict(encoded)
        out["filepath"] = filepath

        return out

    def process_spectrogram(self, processor, audio, label, filepath: str):
        spectrogram = self.spectrogram_fn(audio)

        if self.use_augmentations and self.augmenter is not None:
            spectrogram = self.augmenter.spectral_transform(spectrogram)

        spectrogram_processed = processor(spectrogram)

        return {
            "filepath": filepath,
            "spectrograms": spectrogram_processed,
            "labels": torch.tensor(label),
        }

    def _load_audio(self, filepath: str, idx: int):
        if self.multiply_sample_and_cache_random_window:
            # audio, sr = librosa.load(filepath, sr=self.sample_rate)
            audio, sr = sf.read(filepath)
            assert (
                sr == self.sample_rate
            ), f"Actual sample rate: {sr}, expected: {self.sample_rate}"

            audio = audio[
                self.random_start_indices[idx] : self.random_start_indices[idx]
                + self.max_sequence_length
            ]
        else:
            # audio, sr = librosa.load(
            #     filepath,
            #     sr=self.sample_rate,
            #     duration=self.max_sequence_length / self.sample_rate,
            # )

            audio, sr = sf.read(filepath, frames=self.max_sequence_length)
            assert (
                sr == self.sample_rate
            ), f"Actual sample rate: {sr}, expected: {self.sample_rate}"

        # padding
        if len(audio) < self.max_sequence_length:
            audio = np.pad(
                audio,
                (0, self.max_sequence_length - len(audio)),
                mode="constant",
                constant_values=0,
            )

        return audio.astype(np.float32)

    def __len__(
        self,
    ):
        return len(self.audio_filepaths)

    def __getitem__(self, idx) -> dict:
        filepath = str(self.audio_filepaths[idx])
        audio = self._load_audio(filepath=filepath, idx=idx)

        if self.use_augmentations and self.augmenter is not None:
            audio = self.augmenter.time_transform(audio, sample_rate=self.sample_rate)

        label = self.labels[idx]

        if self.use_spectrograms:
            return self.process_spectrogram(self.processor, audio, label, filepath)
        else:
            return self.process_audio(self.processor, audio, label, filepath)


def get_dataset(filepath: str, processor, augmenter, spectrogram_fn, args):
    return AudioMultiClassDataset(
        filepath=filepath,
        audio_filepart_column=args.audio_filepart_column,
        labels_column=args.labels_column,
        audio_root_dir=args.audio_root_dir,
        processor=processor,
        possible_labels=args.possible_labels,
        max_sequence_length=args.max_sequence_length,
        sample_rate=args.sample_rate,
        drop_duplicates=args.drop_duplicates,
        use_augmentations=args.use_augmentations,
        augmenter=augmenter,
        multiply_sample_and_cache_random_window=args.multiply_sample_and_cache_random_window,
        multiply_factor=args.multiply_factor,
        length_limit=args.length_limit,
        use_spectrograms=args.use_spectrograms,
        spectrogram_fn=spectrogram_fn,
    )
