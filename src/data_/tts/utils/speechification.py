import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, Tuple

import pandas as pd

DEFAULT_DISTRIBUTION_CAN_CONTAIN = (
    "id",
    "name",
    "tts_name",
    "tts_engine",
    "tts_vendor",
)
DEFAULT_DISTRIBUTION_MUST_CONTAIN = DEFAULT_DISTRIBUTION_CAN_CONTAIN[:-1]

DEFAULT_TRANSCRIPTS_MUST_CONTAIN = (
    "id",
    "name",
    "transcript_raw",
)


@dataclass
class SpeechifyArguments:
    DEFAULT_TEXT: ClassVar[Optional[str]] = None
    DEFAULT_DISTRIBUTION_PATH: ClassVar[Optional[Path]] = None
    DEFAULT_TRANSCRIPTS_PATH: ClassVar[Optional[Path]] = None
    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path(".").resolve()
    DEFAULT_OUTPUT_NAME: ClassVar[Optional[str]] = None
    DEFAULT_OVERWRITE: ClassVar[bool] = False
    DEFAULT_VERBOSE: bool = True

    text: Optional[str] = DEFAULT_TEXT
    distribution_path: Optional[Path] = DEFAULT_DISTRIBUTION_PATH
    transcripts_path: Optional[Path] = DEFAULT_TRANSCRIPTS_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_name: str = DEFAULT_OUTPUT_NAME
    overwrite: bool = DEFAULT_OVERWRITE
    verbose: bool = DEFAULT_VERBOSE

    def __post_init__(self):
        self.text = self.DEFAULT_TEXT if self.text is None else str(self.text)
        self.distribution_path = (
            self.DEFAULT_DISTRIBUTION_PATH
            if self.distribution_path is None
            else Path(self.distribution_path).resolve()
        )
        self.transcripts_path = (
            self.DEFAULT_TRANSCRIPTS_PATH
            if self.transcripts_path is None
            else Path(self.transcripts_path).resolve()
        )
        self.output_dir = Path(
            self.DEFAULT_OUTPUT_DIR if self.output_dir is None else self.output_dir
        ).resolve()
        self.output_name = (
            self.DEFAULT_OUTPUT_NAME
            if self.output_name is None
            else str(self.output_name)
        )
        self.overwrite = bool(
            self.DEFAULT_OVERWRITE if self.overwrite is None else self.overwrite
        )
        self.verbose = bool(
            self.DEFAULT_VERBOSE if self.verbose is None else self.verbose
        )

        if self.text is None:
            assert (
                self.distribution_path and self.distribution_path.exists()
            ), f"Distribution path: Expected it to exist, but it doesn't"
            assert (
                self.transcripts_path and self.transcripts_path.exists()
            ), f"Transcripts path: Expected it to exist, but it doesn't"
        else:
            assert self.text, f"Text: Expected a value, got nil"
            assert self.output_name, f"Output name: Expected a value, got nil"

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--text",
            type=str,
            default=SpeechifyArguments.DEFAULT_TEXT,
            metavar="STRING",
            help="The text you wish to synthesize speech for",
        )
        inputs.add_argument(
            "--distribution_path",
            type=str,
            default=SpeechifyArguments.DEFAULT_DISTRIBUTION_PATH,
            metavar="FILE",
            help=(
                "Path to distribution.csv; must have id, name, tts_name and sometimes "
                "tts_engine"
            ),
        )
        inputs.add_argument(
            "--transcripts_path",
            type=str,
            default=SpeechifyArguments.DEFAULT_TRANSCRIPTS_PATH,
            metavar="FILE",
            help="Path to transcripts.csv; must have id, name and transcript_raw",
        )

        outputs.add_argument(
            "--output_dir",
            type=str,
            default=SpeechifyArguments.DEFAULT_OUTPUT_DIR,
            metavar="DIR",
            help=(
                "Path to the output directory. Default: "
                f"{SpeechifyArguments.DEFAULT_OUTPUT_DIR}"
            ),
        )
        outputs.add_argument(
            "--output_name",
            type=str,
            default=SpeechifyArguments.DEFAULT_OUTPUT_NAME,
            metavar="FILENAME",
            help="Filename for the synthesized speech",
        )
        outputs.add_argument(
            "--overwrite",
            action="store_true",
            help="If set, will overwrite existing files, otherwise skip them",
        )
        outputs.add_argument(
            "--verbose",
            action="store_true",
            help="If set, will print out progress, otherwise only warnings and errors",
        )

        return parser

    @staticmethod
    def from_args(args) -> "SpeechifyArguments":
        text = args.text
        distribution_path = args.distribution_path
        transcripts_path = args.transcripts_path
        output_dir = args.output_dir
        output_name = args.output_name
        overwrite = args.overwrite
        verbose = args.verbose

        return SpeechifyArguments(
            text=text,
            distribution_path=distribution_path,
            transcripts_path=transcripts_path,
            output_dir=output_dir,
            output_name=output_name,
            overwrite=overwrite,
            verbose=verbose,
        )


def get_distribution_df(
    distribution_path: Path,
    vendor: str,
    can_contain: Tuple[str, ...] = DEFAULT_DISTRIBUTION_CAN_CONTAIN,
    must_contain: Tuple[str, ...] = DEFAULT_DISTRIBUTION_MUST_CONTAIN,
) -> pd.DataFrame:
    columns = list(pd.read_csv(distribution_path, nrows=0).columns)
    columns = [x for x in can_contain if x in columns]
    for c in must_contain:
        assert c in columns, f"Distribution: Expected column {c}, but it's missing"

    df = pd.read_csv(distribution_path, usecols=columns)
    if "tts_vendor" in df.columns:
        df = df[df["tts_vendor"] == vendor].drop("tts_vendor", axis=1)
    df = (
        df.dropna(how="any", subset=["id", "name"])
        .dropna(how="all", subset=["tts_name", "tts_engine"])
        .drop_duplicates()
        .sort_values(list(df.columns))
        .reset_index(drop=True)
    )

    return df


def get_transcripts_df(
    transcripts_path: Path,
    must_contain: Tuple[str, ...] = DEFAULT_TRANSCRIPTS_MUST_CONTAIN,
) -> pd.DataFrame:
    columns = list(pd.read_csv(transcripts_path, nrows=0).columns)
    columns = [x for x in must_contain if x in columns]
    for c in must_contain:
        assert c in columns, f"Transcripts: Expected column {c}, but it's missing"

    df = pd.read_csv(transcripts_path, usecols=columns)
    df = (
        df.dropna()
        .drop_duplicates()
        .sort_values(list(df.columns))
        .reset_index(drop=True)
    )

    return df


def get_speechify_df(
    distribution_df: pd.DataFrame, transcripts_df: pd.DataFrame
) -> pd.DataFrame:
    df = pd.merge(
        transcripts_df,
        distribution_df,
        how="left",
        left_on=["id", "name"],
        right_on=["id", "name"],
    ).drop("id", axis=1)
    df = df.drop_duplicates().sort_values(list(df.columns)).reset_index(drop=True)

    return df
