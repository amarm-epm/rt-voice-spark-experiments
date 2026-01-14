"""
STT wrapper using pywhispercpp (whisper.cpp with CUDA).
"""

import tempfile
import wave
from pathlib import Path

import numpy as np
from pywhispercpp.model import Model

WHISPER_SAMPLE_RATE = 16000


def resample_wav(input_path: str, target_rate: int = WHISPER_SAMPLE_RATE) -> str:
    """Resample WAV file to target sample rate if needed."""
    with wave.open(input_path, "rb") as wav_in:
        orig_rate = wav_in.getframerate()
        n_channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        n_frames = wav_in.getnframes()

        if orig_rate == target_rate:
            return input_path

        # Read audio data
        audio_data = wav_in.readframes(n_frames)

    # Convert to numpy array
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        dtype = np.uint8

    audio_array = np.frombuffer(audio_data, dtype=dtype)

    # Handle stereo -> mono
    if n_channels == 2:
        audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(dtype)

    # Resample using linear interpolation
    duration = len(audio_array) / orig_rate
    new_length = int(duration * target_rate)
    indices = np.linspace(0, len(audio_array) - 1, new_length)
    resampled = np.interp(indices, np.arange(len(audio_array)), audio_array.astype(np.float32))
    resampled = resampled.astype(dtype)

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_file.name, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(sampwidth)
        wav_out.setframerate(target_rate)
        wav_out.writeframes(resampled.tobytes())

    return temp_file.name


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
            audio_path: Path to audio file (WAV format, any sample rate)

        Returns:
            Transcribed text
        """
        # Resample to 16kHz if needed
        audio_path = resample_wav(str(audio_path))

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
