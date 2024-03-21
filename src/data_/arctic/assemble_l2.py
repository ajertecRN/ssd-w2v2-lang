import argparse
from dataclasses import dataclass
import glob
from pathlib import Path
import shutil

from utils import arctic_rows_to_df


@dataclass
class AssembleL2Arguments:
    l2_dir: Path
    result_dir: Path

    def __post_init__(self):
        self.l2_dir = Path(self.l2_dir).resolve()
        self.result_dir = Path(self.result_dir).resolve()

        assert self.l2_dir.exists(), f"L2 directory: Expected it to exist"

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--l2_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the L2 speaker directory (ex. ABA)",
        )

        outputs.add_argument(
            "--result_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the directory where results will be saved",
        )

        return parser

    @staticmethod
    def from_args(args) -> "AssembleL2Arguments":
        l2_dir = args.l2_dir
        result_dir = args.result_dir

        return AssembleL2Arguments(l2_dir=l2_dir, result_dir=result_dir)


def assemble_l2(arguments: AssembleL2Arguments):
    data_dir = arguments.result_dir / "data"
    audio_dir = arguments.result_dir / "audio" / "original"

    data_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = arguments.l2_dir / "transcript"
    prompts = glob.glob(f"{prompts_dir}/arctic_*.txt")
    prompts = [Path(prompt) for prompt in prompts]
    prompts = [prompt for prompt in prompts if prompt.is_file()]
    prompts = sorted(prompts)

    rows = list()
    for prompt in prompts:
        with open(prompt, mode="r", encoding="utf8", errors="replace") as f:
            name = prompt.name
            i = name.split(".")[0]
            transcript = f.read()

            rows.append((i, name, transcript))
    arctic_rows_to_df(rows=rows).to_csv(data_dir / "transcripts.csv", index=False)

    utterance_dir = arguments.l2_dir / "wav"
    utterance_files = glob.glob(f"{utterance_dir}/*.wav")
    utterance_files = [Path(utterance) for utterance in utterance_files]
    utterance_files = [
        utterance for utterance in utterance_files if utterance.is_file()
    ]
    utterance_files = sorted(utterance_files)
    for utterance in utterance_files:
        shutil.copyfile(utterance, audio_dir / utterance.name)


def main():
    parser = AssembleL2Arguments.get_parser()
    args = parser.parse_args()
    arguments = AssembleL2Arguments.from_args(args=args)

    assemble_l2(arguments=arguments)


if __name__ == "__main__":
    main()
