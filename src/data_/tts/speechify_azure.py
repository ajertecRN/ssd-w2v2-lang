from dataclasses import asdict, dataclass
import sys
from typing import ClassVar

from tqdm import tqdm

from azure.synthesizer import (
    AZURE_OUTPUT_FORMAT_STRING_TO_ENUM,
    AzureSynthesizer,
    AzureSynthesizerConfig,
)
from utils.speechification import (
    SpeechifyArguments,
    get_distribution_df,
    get_speechify_df,
    get_transcripts_df,
)

AZURE_DISTRIBUTION_CAN_CONTAIN = (
    "id",
    "name",
    "tts_name",
    "tts_vendor",
)
AZURE_DISTRIBUTION_MUST_CONTAIN = AZURE_DISTRIBUTION_CAN_CONTAIN[:-1]


@dataclass
class SpeechifyAzureArguments(SpeechifyArguments):
    DEFAULT_VOICE_NAME: ClassVar[str] = AzureSynthesizerConfig.DEFAULT_VOICE_NAME
    DEFAULT_OUTPUT_FORMAT: ClassVar[str] = AzureSynthesizerConfig.DEFAULT_OUTPUT_FORMAT

    voice_name: str = DEFAULT_VOICE_NAME
    output_format: str = DEFAULT_OUTPUT_FORMAT

    def __post_init__(self):
        self.voice_name = str(
            self.DEFAULT_VOICE_NAME if self.voice_name is None else self.voice_name
        )
        self.output_format = str(
            self.DEFAULT_OUTPUT_FORMAT
            if self.output_format is None
            else self.output_format
        )

    @staticmethod
    def get_parser():
        parser = super(SpeechifyAzureArguments, SpeechifyAzureArguments).get_parser()

        tts_group = parser.add_argument_group("TTS")

        tts_group.add_argument(
            "--voice_name",
            type=str,
            default=SpeechifyAzureArguments.DEFAULT_VOICE_NAME,
            metavar="STRING",
            help=(
                "Name of the TTS voice for synthesizing speech. Default: "
                f"{SpeechifyAzureArguments.DEFAULT_VOICE_NAME}"
            ),
        )
        tts_group.add_argument(
            "--output_format",
            type=str,
            choices=sorted(AZURE_OUTPUT_FORMAT_STRING_TO_ENUM.keys()),
            default=SpeechifyAzureArguments.DEFAULT_OUTPUT_FORMAT,
            metavar="STRING",
            help=(
                "Audio format of the synthesized speech. Default: "
                f"{SpeechifyAzureArguments.DEFAULT_OUTPUT_FORMAT}"
            ),
        )

        return parser

    @staticmethod
    def from_args(args) -> "SpeechifyAzureArguments":
        base_args = super(SpeechifyAzureArguments, SpeechifyAzureArguments).from_args(
            args
        )
        voice_name = args.voice_name
        output_format = args.output_format

        return SpeechifyAzureArguments(
            **asdict(base_args),
            voice_name=voice_name,
            output_format=output_format,
        )


def speechify_azure(arguments: SpeechifyAzureArguments):
    if arguments.text:
        synthesizer = AzureSynthesizer(
            voice_name=arguments.voice_name,
            output_format=arguments.output_format,
        )

        output_path = arguments.output_dir / arguments.output_name
        if output_path.exists() and not arguments.overwrite:
            print(
                f"Output path already exists, but overwrite is disabled",
                file=sys.stderr,
            )
        else:
            result = synthesizer.synthesize(text=arguments.text)
            result.save(output_path)
    else:
        df = get_speechify_df(
            distribution_df=get_distribution_df(
                distribution_path=arguments.distribution_path,
                vendor="azure",
                can_contain=AZURE_DISTRIBUTION_CAN_CONTAIN,
                must_contain=AZURE_DISTRIBUTION_MUST_CONTAIN,
            ),
            transcripts_df=get_transcripts_df(
                transcripts_path=arguments.transcripts_path
            ),
        )

        iterable = df[["name", "transcript_raw", "tts_name"]].itertuples(index=False)
        if arguments.verbose:
            iterable = tqdm(
                iterable,
                desc="Azure TTS",
                total=df.shape[0],
                file=sys.stdout,
                ncols=80,
                unit="sent",
                unit_scale=True,
            )
        for name, transcript, voice_name in iterable:
            synthesizer = AzureSynthesizer(
                voice_name=voice_name,
                output_format=arguments.output_format,
            )

            output_filename = str(name).split(".")[0] + f".{synthesizer.extension}"
            output_path = arguments.output_dir / output_filename
            if output_path.exists() and not arguments.overwrite:
                continue

            result = synthesizer.synthesize(text=transcript)
            result.save(output_path)


def main():
    parser = SpeechifyAzureArguments.get_parser()
    args = parser.parse_args()
    arguments = SpeechifyAzureArguments.from_args(args=args)

    speechify_azure(arguments=arguments)


if __name__ == "__main__":
    main()
