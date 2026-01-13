"""
STT wrapper using pywhispercpp (whisper.cpp with CUDA).
"""

from pathlib import Path
from pywhispercpp.model import Model


class WhisperSTT:
    """Wrapper for Whisper speech-to-text with CUDA acceleration."""

    def __init__(
        self,
        model_size: str = "base.en",
        n_threads: int = 4,
    ):
        """
        Initialize Whisper model.

        Args:
            model_size: Model size (tiny.en, base.en, small.en, medium.en, large-v3)
            n_threads: Number of CPU threads for decoding
        """
        self.model = Model(model_size, n_threads=n_threads)
        self.model_size = model_size

    def transcribe(self, audio_path: Path | str) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (WAV format, 16kHz recommended)

        Returns:
            Transcribed text
        """
        segments = self.model.transcribe(str(audio_path))
        text = " ".join([seg.text.strip() for seg in segments])
        return text


def test_stt(audio_path: str = None):
    """Quick test of STT functionality."""
    import time

    if audio_path is None:
        audio_path = "tmp/whisper.cpp/samples/jfk.wav"

    print("Loading Whisper model (CUDA)...")
    t0 = time.time()
    stt = WhisperSTT()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    print(f"Transcribing: {audio_path}")
    t0 = time.time()
    text = stt.transcribe(audio_path)
    transcribe_time = time.time() - t0

    print(f"Transcript: {text}")
    print()
    print("=" * 40)
    print("METRICS")
    print("=" * 40)
    print(f"Load time: {load_time:.2f}s")
    print(f"Transcribe time: {transcribe_time:.2f}s")

    return text


if __name__ == "__main__":
    import sys
    audio = sys.argv[1] if len(sys.argv) > 1 else None
    test_stt(audio)
