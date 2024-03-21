import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import ClassVar

import pandas as pd
from tqdm import tqdm

from google_.synthesizer import GoogleSynthesizer, GoogleSynthesizerConfig


@dataclass
class SpeechifyARCTICArguments:
    DEFAULT_OVERWRITE: ClassVar[bool] = False
    DEFAULT_VERBOSE: ClassVar[bool] = False

    transcripts_path: Path
    voices_path: Path
    result_dir: Path
    overwrite: bool = DEFAULT_OVERWRITE
    verbose: bool = DEFAULT_VERBOSE

    def __post_init__(self):
        self.transcripts_path = Path(self.transcripts_path).resolve()
        self.voices_path = Path(self.voices_path).resolve()
        self.result_dir = Path(self.result_dir).resolve()
        self.overwrite = bool(
            self.DEFAULT_OVERWRITE if self.overwrite is None else self.overwrite
        )
        self.verbose = bool(
            self.DEFAULT_VERBOSE if self.verbose is None else self.verbose
        )

        assert self.transcripts_path.exists(), f"Transcripts path: Expected it to exist"
        assert self.voices_path.exists(), f"Voices path: Expected it to exist"

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--transcripts_path",
            type=str,
            required=True,
            metavar="FILEPATH",
            help="Path to the transcripts CSV file",
        )
        inputs.add_argument(
            "--voices_path",
            type=str,
            required=True,
            metavar="FILEPATH",
            help="Path to the voices CSV file",
        )

        outputs.add_argument(
            "--result_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the root directory where results will be saved (in folders)",
        )
        outputs.add_argument(
            "--overwrite",
            action="store_true",
            help=(
                "If set, will overwrite on filename collision, otherwise skip the "
                "sample"
            ),
        )
        outputs.add_argument(
            "--verbose",
            action="store_true",
            help="If set, will output progress",
        )

        return parser

    @staticmethod
    def from_args(args):
        transcripts_path = args.transcripts_path
        voices_path = args.voices_path
        result_dir = args.result_dir
        overwrite = args.overwrite
        verbose = args.verbose

        return SpeechifyARCTICArguments(
            transcripts_path=transcripts_path,
            voices_path=voices_path,
            result_dir=result_dir,
            overwrite=overwrite,
            verbose=verbose,
        )


def speechify_arctic(arguments: SpeechifyARCTICArguments):
    transcripts = (
        pd.read_csv(arguments.transcripts_path)
        .dropna()
        .drop_duplicates()
        .sort_values(["id", "name", "transcript"])
        .reset_index(drop=True)
    )
    voices = pd.read_csv(arguments.voices_path, usecols=["tts_vendor", "tts_name"])
    voices = (
        voices[voices["tts_vendor"] == "google"]
        .drop("tts_vendor", axis=1)
        .dropna()
        .drop_duplicates()["tts_name"]
        .to_list()
    )
    voices = sorted(voices)

    for voice in voices:
        current_dir = arguments.result_dir / str(voice).lower()
        audio_dir = current_dir / "audio"
        data_dir = current_dir / "data"
        audio_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        transcripts.to_csv(data_dir / "transcripts.csv", index=False)

        synthesizer = GoogleSynthesizer(voice_name=voice)

        iterator = transcripts.itertuples(index=False)
        if arguments.verbose:
            iterator = tqdm(
                iterator,
                desc=f"ARCTIC {voice}",
                total=transcripts.shape[0],
                file=sys.stdout,
                ncols=80,
                unit="sent",
                unit_scale=True,
            )
        for i, name, transcript in iterator:
            result_path = data_dir / name

            if not arguments.overwrite and result_path.exists():
                continue

            result = synthesizer.synthesize(text=transcript)
            result.save(data_dir / name)


def main():
    parser = SpeechifyARCTICArguments.get_parser()
    args = parser.parse_args()
    arguments = SpeechifyARCTICArguments.from_args(args=args)

    speechify_arctic(arguments=arguments)


if __name__ == "__main__":
    main()
