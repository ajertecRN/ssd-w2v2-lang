import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from utils import assemble_dataset as assemble_dataset_prime, FLEURS_LANGUAGE_CODES


@dataclass
class AssembleDatasetArguments:
    DEFAULT_LANGUAGE_CODE: ClassVar[str] = "en_us"
    DEFAULT_CLEAN_UP: ClassVar[bool] = False

    result_dir: Path
    language_code: str = DEFAULT_LANGUAGE_CODE
    clean_up: bool = DEFAULT_CLEAN_UP

    def __post_init__(self):
        self.result_dir = Path(self.result_dir).resolve()
        self.language_code = (
            str(
                self.DEFAULT_LANGUAGE_CODE
                if self.language_code is None
                else self.language_code
            )
            .strip()
            .lower()
        )
        self.clean_up = bool(
            self.DEFAULT_CLEAN_UP if self.clean_up is None else self.clean_up
        )

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--language_code",
            type=str,
            choices=FLEURS_LANGUAGE_CODES,
            default=AssembleDatasetArguments.DEFAULT_LANGUAGE_CODE,
            metavar="STRING",
            help=(
                "Language code of the dataset you want to assemble. Default: "
                f"{AssembleDatasetArguments.DEFAULT_LANGUAGE_CODE}"
            ),
        )

        outputs.add_argument(
            "--result_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the directory where the dataset will be assembled",
        )
        outputs.add_argument(
            "--clean_up",
            action="store_true",
            help=(
                "If set, will clean up the cache files. WARNING: If you do this, "
                "rerunning the script will require redownloading the files. It's "
                "recommended you delete the cache files separately, this serves only "
                "as convenience"
            ),
        )

        return parser

    @staticmethod
    def from_args(args):
        language_code = args.language_code
        result_dir = args.result_dir
        clean_up = args.clean_up

        return AssembleDatasetArguments(
            result_dir=result_dir, language_code=language_code, clean_up=clean_up
        )


def main():
    parser = AssembleDatasetArguments.get_parser()
    args = parser.parse_args()
    arguments = AssembleDatasetArguments.from_args(args=args)

    assemble_dataset_prime(
        dataset_dir=arguments.result_dir,
        language_code=arguments.language_code,
        clean_up=arguments.clean_up,
    )


if __name__ == "__main__":
    main()
