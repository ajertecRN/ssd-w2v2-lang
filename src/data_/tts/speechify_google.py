from dataclasses import asdict, dataclass
import sys
from typing import ClassVar

from tqdm import tqdm

from google_.synthesizer import (
    GOOGLE_AUDIO_ENCODING_STRING_TO_ENUM,
    GoogleSynthesizer,
    GoogleSynthesizerConfig,
)
from utils.speechification import (
    SpeechifyArguments,
    get_distribution_df,
    get_speechify_df,
    get_transcripts_df,
)

GOOGLE_DISTRIBUTION_CAN_CONTAIN = (
    "id",
    "name",
    "tts_name",
    "tts_vendor",
)
GOOGLE_DISTRIBUTION_MUST_CONTAIN = GOOGLE_DISTRIBUTION_CAN_CONTAIN[:-1]


@dataclass
class SpeechifyGoogleArguments(SpeechifyArguments):
    DEFAULT_AUDIO_ENCODING: ClassVar[
        str
    ] = GoogleSynthesizerConfig.DEFAULT_AUDIO_ENCODING
    DEFAULT_SAMPLE_RATE: ClassVar[int] = GoogleSynthesizerConfig.DEFAULT_SAMPLE_RATE
    DEFAULT_VOICE_NAME: ClassVar[str] = GoogleSynthesizerConfig.DEFAULT_VOICE_NAME

    audio_encoding: str = DEFAULT_AUDIO_ENCODING
    sample_rate: int = DEFAULT_SAMPLE_RATE
    voice_name: str = DEFAULT_VOICE_NAME

    def __post_init__(self):
        self.audio_encoding = str(
            self.DEFAULT_AUDIO_ENCODING
            if self.audio_encoding is None
            else self.audio_encoding
        )
        self.sample_rate = int(
            self.DEFAULT_SAMPLE_RATE if self.sample_rate is None else self.sample_rate
        )
        self.voice_name = str(
            self.DEFAULT_VOICE_NAME if self.voice_name is None else self.voice_name
        )

    @staticmethod
    def get_parser():
        parser = super(SpeechifyGoogleArguments, SpeechifyGoogleArguments).get_parser()

        tts_group = parser.add_argument_group("TTS")

        tts_group.add_argument(
            "--audio_encoding",
            type=str,
            choices=sorted(GOOGLE_AUDIO_ENCODING_STRING_TO_ENUM.keys()),
            default=SpeechifyGoogleArguments.DEFAULT_AUDIO_ENCODING,
            metavar="STRING",
            help=(
                "Audio encoding of the synthesized speech. Default: "
                f"{SpeechifyGoogleArguments.DEFAULT_AUDIO_ENCODING}"
            ),
        )
        tts_group.add_argument(
            "--sample_rate",
            type=int,
            default=SpeechifyGoogleArguments.DEFAULT_SAMPLE_RATE,
            metavar="HERTZ",
            help=(
                "Sample rate of the synthesized speech. Default: "
                f"{SpeechifyGoogleArguments.DEFAULT_SAMPLE_RATE}"
            ),
        )
        tts_group.add_argument(
            "--voice_name",
            type=str,
            default=SpeechifyGoogleArguments.DEFAULT_VOICE_NAME,
            metavar="STRING",
            help=(
                "Name of the TTS voice for synthesizing speech. Default: "
                f"{SpeechifyGoogleArguments.DEFAULT_VOICE_NAME}"
            ),
        )

        return parser

    @staticmethod
    def from_args(args) -> "SpeechifyGoogleArguments":
        base_args = super(SpeechifyGoogleArguments, SpeechifyGoogleArguments).from_args(
            args
        )
        audio_encoding = args.audio_encoding
        sample_rate = args.sample_rate
        voice_name = args.voice_name

        return SpeechifyGoogleArguments(
            **asdict(base_args),
            audio_encoding=audio_encoding,
            sample_rate=sample_rate,
            voice_name=voice_name,
        )


def speechify_google(arguments: SpeechifyGoogleArguments):
    if arguments.text:
        synthesizer = GoogleSynthesizer(
            audio_encoding=arguments.audio_encoding,
            sample_rate=arguments.sample_rate,
            voice_name=arguments.voice_name,
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
                vendor="google",
                can_contain=GOOGLE_DISTRIBUTION_CAN_CONTAIN,
                must_contain=GOOGLE_DISTRIBUTION_MUST_CONTAIN,
            ),
            transcripts_df=get_transcripts_df(
                transcripts_path=arguments.transcripts_path
            ),
        )

        iterable = df[["name", "transcript_raw", "tts_name"]].itertuples(index=False)
        if arguments.verbose:
            iterable = tqdm(
                iterable,
                desc="Amazon TTS",
                total=df.shape[0],
                file=sys.stdout,
                ncols=80,
                unit="sent",
                unit_scale=True,
            )
        for name, transcript, voice_name in iterable:
            synthesizer = GoogleSynthesizer(
                audio_encoding=arguments.audio_encoding,
                sample_rate=arguments.sample_rate,
                voice_name=voice_name,
            )
            output_filename = str(name).split(".")[0] + f".{synthesizer.extension}"
            output_path = arguments.output_dir / output_filename

            if output_path.exists() and not arguments.overwrite:
                continue

            result = synthesizer.synthesize(text=transcript)
            result.save(output_path)


def main():
    parser = SpeechifyGoogleArguments.get_parser()
    args = parser.parse_args()
    arguments = SpeechifyGoogleArguments.from_args(args=args)

    speechify_google(arguments=arguments)


if __name__ == "__main__":
    main()
