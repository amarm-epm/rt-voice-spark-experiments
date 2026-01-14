"""
TTS wrapper using Piper.
"""

import wave
from pathlib import Path
from piper import PiperVoice


# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "piper" / "en_US-lessac-medium.onnx"


class PiperTTS:
    """Wrapper for Piper text-to-speech."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        use_cuda: bool = True,
    ):
        """
        Initialize Piper TTS.

        Args:
            model_path: Path to ONNX model file
            use_cuda: Whether to use CUDA acceleration
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.voice = PiperVoice.load(str(self.model_path), use_cuda=use_cuda)

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of the voice."""
        return self.voice.config.sample_rate

    def synthesize(self, text: str, output_path: Path | str) -> Path:
        """
        Synthesize text to audio file.

        Args:
            text: Text to synthesize
            output_path: Path to output WAV file

        Returns:
            Path to output file
        """
        output_path = Path(output_path)

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            for audio_chunk in self.voice.synthesize(text):
                wav_file.writeframes(audio_chunk.audio_int16_bytes)

        return output_path

    def synthesize_stream(self, text: str):
        """
        Synthesize text and yield audio chunks.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as bytes
        """
        for audio_bytes in self.voice.synthesize_stream_raw(text):
            yield audio_bytes


def test_tts(text: str = None, output_path: str = None):
    """Quick test of TTS functionality."""
    import time

    if text is None:
        text = "Hello! This is a test of the Piper text to speech system."
    if output_path is None:
        output_path = "tmp/tts_output.wav"

    print("Loading Piper TTS...")
    t0 = time.time()
    tts = PiperTTS()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    print(f"Synthesizing: {text}")
    t0 = time.time()
    tts.synthesize(text, output_path)
    synth_time = time.time() - t0

    print(f"Output saved to: {output_path}")
    print()
    print("=" * 40)
    print("METRICS")
    print("=" * 40)
    print(f"Load time: {load_time:.2f}s")
    print(f"Synthesis time: {synth_time:.2f}s")
    print(f"Text length: {len(text)} chars")

    return output_path


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else None
    output = sys.argv[2] if len(sys.argv) > 2 else None
    test_tts(text, output)
