"""
STT wrapper using faster-whisper.
"""

from pathlib import Path
from faster_whisper import WhisperModel


def _detect_device():
    """Auto-detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "float32"


class WhisperSTT:
    """Wrapper for Whisper speech-to-text."""

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        compute_type: str | None = None,
    ):
        """
        Initialize Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (cuda, cpu, auto). None = auto-detect
            compute_type: Compute type (float16, int8, float32). None = auto
        """
        if device is None or compute_type is None:
            auto_device, auto_compute = _detect_device()
            device = device or auto_device
            compute_type = compute_type or auto_compute

        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "en",
    ) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., "en")

        Returns:
            Transcribed text
        """
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
        )

        # Combine all segments into single text
        text = " ".join(segment.text.strip() for segment in segments)
        return text


def test_stt(audio_path: str):
    """Quick test of STT functionality."""
    print("Loading Whisper model (auto-detect device)...")
    stt = WhisperSTT()
    print(f"Using device: {stt.device}, compute_type: {stt.compute_type}")

    print(f"Transcribing: {audio_path}")
    text = stt.transcribe(audio_path)
    print(f"Transcript: {text}")

    return text


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_stt(sys.argv[1])
    else:
        print("Usage: python -m src.stt.whisper_stt <audio_file>")
