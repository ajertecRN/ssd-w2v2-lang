from dataclasses import dataclass
from pathlib import Path
import sys
from typing import ClassVar, Optional

from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing

from .constants import (
    AMAZON_ENGINES,
    AMAZON_OUTPUT_FORMATS,
    AMAZON_OUTPUT_FORMAT_TO_EXTENSION,
)


@dataclass
class AmazonSynthesizerConfig:
    DEFAULT_ENGINE: ClassVar[str] = "standard"
    DEFAULT_OUTPUT_FORMAT: ClassVar[str] = "pcm"
    DEFAULT_SAMPLE_RATE: ClassVar[int] = 16000
    DEFAULT_VOICE_NAME: ClassVar[str] = "Joanna"
    DEFAULT_PROFILE_NAME: ClassVar[str] = "default"

    engine: str = DEFAULT_ENGINE
    output_format: str = DEFAULT_OUTPUT_FORMAT
    sample_rate: int = DEFAULT_SAMPLE_RATE
    voice_name: str = DEFAULT_VOICE_NAME
    profile_name: str = DEFAULT_PROFILE_NAME

    def __post_init__(self):
        self.engine = (
            str(self.DEFAULT_ENGINE if self.engine is None else self.engine)
            .strip()
            .lower()
        )
        self.output_format = (
            str(
                self.DEFAULT_OUTPUT_FORMAT
                if self.output_format is None
                else self.output_format
            )
            .strip()
            .lower()
        )
        self.sample_rate = int(
            self.DEFAULT_SAMPLE_RATE if self.sample_rate is None else self.sample_rate
        )
        self.voice_name = (
            str(self.DEFAULT_VOICE_NAME if self.voice_name is None else self.voice_name)
            .strip()
            .capitalize()
        )
        self.profile_name = str(
            self.DEFAULT_PROFILE_NAME
            if self.profile_name is None
            else self.profile_name
        ).strip()

        assert self.engine in AMAZON_ENGINES, f"Engine: Unsupported ({self.engine})"
        assert (
            self.output_format in AMAZON_OUTPUT_FORMATS
        ), f"Output format: Unsupported ({self.output_format})"
        assert (
            self.sample_rate > 0
        ), f"Sample rate: Expected > 0, got {self.sample_rate}"
        assert self.voice_name, f"Voice name: Expected a value, got nil"
        assert self.profile_name, f"Profile name: Expected a value, got nil"

    @property
    def extension(self) -> str:
        return AMAZON_OUTPUT_FORMAT_TO_EXTENSION[self.output_format]

    def to_args(self):
        return {
            "Engine": self.engine,
            "OutputFormat": self.output_format,
            "SampleRate": str(self.sample_rate),
            "TextType": "text",
            "VoiceId": self.voice_name,
        }

    def get_args(self, text: str):
        text = str("" if text is None else text)

        args = self.to_args()
        args["Text"] = text

        return args


@dataclass
class AmazonSynthesizerResult:
    DEFAULT_DATA: ClassVar[bytes] = b""
    DEFAULT_MESSAGE: ClassVar[str] = ""

    data: bytes = DEFAULT_DATA
    message: str = DEFAULT_MESSAGE

    def __post_init__(self):
        self.data = bytes(self.DEFAULT_DATA if self.data is None else self.data)
        self.message = str(
            self.DEFAULT_MESSAGE if self.message is None else self.message
        ).strip()

        assert self.message or self.data, f"Expected some data or a message"

    def save(self, path: Path):
        if self.data:
            with open(path, mode="wb+") as f:
                f.write(self.data)
        else:
            print(f"[{path.name}] {self.message}", file=sys.stderr)


class AmazonSynthesizer:
    def __init__(
        self,
        engine: Optional[str] = None,
        output_format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        voice_name: Optional[str] = None,
        profile_name: Optional[str] = None,
    ):
        self._config = AmazonSynthesizerConfig(
            engine=engine,
            output_format=output_format,
            sample_rate=sample_rate,
            voice_name=voice_name,
            profile_name=profile_name,
        )

        import os

        self._session = Session(profile_name=self.profile_name)
        self._client = self.session.client("polly")

    # region Properties
    @property
    def config(self) -> AmazonSynthesizerConfig:
        return self._config

    @property
    def engine(self) -> str:
        return self.config.engine

    @property
    def output_format(self) -> str:
        return self.config.output_format

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def voice_name(self) -> str:
        return self.config.voice_name

    @property
    def profile_name(self) -> str:
        return self.config.profile_name

    @property
    def extension(self) -> str:
        return self.config.extension

    @property
    def session(self) -> Session:
        return self._session

    @property
    def client(self):
        return self._client

    # endregion

    def synthesize(self, text: str) -> AmazonSynthesizerResult:
        kwargs = self.config.get_args(text=text)

        try:
            response = self.client.synthesize_speech(**kwargs)
        except (BotoCoreError, ClientError) as error:
            return AmazonSynthesizerResult(message=f"Error: {str(error).strip()}")

        if "AudioStream" not in response:
            return AmazonSynthesizerResult(
                message=f"Error: No AudioStream in response",
            )

        with closing(response["AudioStream"]) as stream:
            data = bytes(stream.read())

        return AmazonSynthesizerResult(data=data)
