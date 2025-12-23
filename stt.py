import os
import time
import queue
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import webrtcvad
import torch
import whisper
from TTS.api import TTS

RATE = 16000 # Sampling rate
FRAME_MS = 30 # Frame size in ms
FRAME_SIZE = int(RATE * FRAME_MS / 1000) # in samples
SILENCE_LIMIT = 2.5 # seconds
AUDIO_FILE = "input.wav" # Temporary audio file
SPEAKER_WAV = "sample.wav"  # voice reference


device = "cuda" if torch.cuda.is_available() else "cpu" # Determine device
print("Using device:", device) # Print device info

# =========================
# LOAD MODELS
whisper_model = whisper.load_model("small", device=device)

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True
)
tts.to(device) # Move TTS model to device

# =========================
# VAD SETUP
# =========================
vad = webrtcvad.Vad(2) # Aggressiveness mode (0-3)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status): # Audio callback function
    audio_queue.put(bytes(indata))

# =========================
# RECORD UNTIL SILENCE
# =========================
def record_until_silence(filename=AUDIO_FILE): # Record audio until silence is detected
    print("🎙️ Listening...")

    voiced_frames = []
    silence_start = None

    with sd.RawInputStream(
        samplerate=RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:
            frame = audio_queue.get()
            is_speech = vad.is_speech(frame, RATE)

            if is_speech:
                voiced_frames.append(frame)
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_LIMIT:
                    break

    audio_bytes = b"".join(voiced_frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    wav.write(filename, RATE, audio_np)

    print("🛑 Speech ended")
    return filename

# =========================
# MAIN LOOP
# =========================
try:
    while True:
        audio_path = record_until_silence()

        result = whisper_model.transcribe(
            audio_path,
            fp16=(device == "cuda")
        )

        text = result["text"].strip()
        print("🗣 You said:", text)

        if not text:
            continue

        if "stop" in text.lower():
            print("👋 Keyword detected. Exiting.")
            break

        tts.tts_to_file(
            text=text,
            file_path="output.wav",
            speaker_wav=SPEAKER_WAV,
            language="en"
        )

        # Play TTS output
        fs, audio = wav.read("output.wav")
        sd.play(audio, fs)
        sd.wait()

except KeyboardInterrupt:
    print("\n🛑 Manually stopped.")
