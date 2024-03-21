from dataclasses import dataclass
import os
from pathlib import Path
from typing import ClassVar, Optional


@dataclass
class CoquiSynthesizerConfig:
    DEFAULT_MODEL_NAME: ClassVar[str] = "tts_models/en/ljspeech/vits"
    DEFAULT_SPEAKER: ClassVar[Optional[str]] = None
    DEFAULT_TTS_DIR: ClassVar[Path] = (
        Path(os.path.expanduser("~")).resolve() / "rn-samples" / "coqui-models"
    )
    DEFAULT_OUTPUT_DIR: ClassVar[Path] = (
        Path(os.path.expanduser("~")).resolve() / "rn-samples" / "coqui-outputs"
    )

    model_name: str = DEFAULT_MODEL_NAME
    speaker: Optional[str] = DEFAULT_SPEAKER
    tts_dir: Path = DEFAULT_TTS_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    _actual_output_dir: ClassVar[Path] = None

    def __post_init__(self):
        self.model_name = str(
            self.DEFAULT_MODEL_NAME if self.model_name is None else self.model_name
        ).strip()
        self.speaker = (
            self.DEFAULT_SPEAKER if self.speaker is None else str(self.speaker)
        )
        self.tts_dir = Path(
            self.DEFAULT_TTS_DIR if self.tts_dir is None else self.tts_dir
        ).resolve()
        self.tts_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(
            self.DEFAULT_OUTPUT_DIR if self.output_dir is None else self.output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        assert self.model_name, f"Model name: Expected a value, got nil"
        assert self.tts_dir.exists(), f"TTS dir: Expected it to exist, but mkdir failed"
        assert (
            self.output_dir.exists()
        ), f"Output dir: Expected it to exist, but mkdir failed"

        self._actual_output_dir = self.output_dir / "/".join(
            self.model_name.split("/")[1:]
        )

    @property
    def actual_output_dir(self) -> Path:
        return self._actual_output_dir


class CoquiSynthesizer:
    def __init__(
        self,
        model_name: Optional[str] = None,
        speaker: Optional[str] = None,
        tts_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self._config = CoquiSynthesizerConfig(
            model_name=model_name,
            speaker=speaker,
            tts_dir=tts_dir,
            output_dir=output_dir,
        )

        self._docker_lines = [
            "sudo docker run",
            "--rm",
            f"-v '{self.tts_dir}:/root/.local/share/tts'",
            f"-v '{self.output_dir}:/root/tts-output'",
        ]
        self._coqui_lines = [
            f"--model_name '{self.model_name}'",
        ]
        if self.speaker:
            self._coqui_lines.append(f"--speaker_idx '{self.speaker}'")

    # region Properties
    @property
    def config(self) -> CoquiSynthesizerConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def speaker(self) -> Optional[str]:
        return self.config.speaker

    @property
    def tts_dir(self) -> Path:
        return self.config.tts_dir

    @property
    def output_dir(self) -> Path:
        return self.config.actual_output_dir

    # endregion

    def synthesize(
        self,
        text: str,
        relative_path: Path,
        use_gpu: bool = False,
        verbose: bool = False,
    ):
        text = str("" if text is None else text).replace("'", "'\\''")
        absolute_path = (self.output_dir / relative_path).resolve()
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        if not absolute_path.parent.exists():
            return "Failed to create destination parent"

        absolute_path = str(absolute_path).replace("'", "'\\''")
        tts_path = "ghcr.io/coqui-ai/tts"
        if not use_gpu:
            tts_path += "-cpu"

        coqui_lines = [
            tts_path,
            *self._coqui_lines,
            f"--text '{text}'",
            f"--out_path '{absolute_path}'",
        ]
        command = " ".join(self._docker_lines + coqui_lines).strip()
        if not verbose:
            command += " > /dev/null 2>&1"

        os.system(command)
