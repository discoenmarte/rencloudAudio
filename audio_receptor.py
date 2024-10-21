import numpy as np
import webrtcvad
import sounddevice as sd
import wave
from scipy.fftpack import fft
from io import BytesIO
import asyncio  
import requests


# Initialize VAD
vad = webrtcvad.Vad(3)  # Sensitivity (0-3). 2 is a good middle ground.

# Audio buffer configuration
audio_buffer = []
is_talking = False  # Track whether the person is talking or not
silence_counter = 0  # Counter to track the amount of consecutive silence frames
SILENCE_FRAMES_THRESHOLD = 30  # Number of silent frames to wait before deciding that speaking has stopped (300ms for 10ms frames)

# Function to convert audio buffer to WAV format
def buffer_to_wav(buffer, sample_rate=16000):
    output = BytesIO()
    with wave.open(output, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(buffer))
    return output.getvalue()

# Function to send audio to Whisper API and get transcription
def send_audio_to_whisper(buffer):
    audio_data = buffer_to_wav(buffer)  # Convert buffer to WAV format
    files = {'file': ('audio.wav', audio_data, 'audio/wav')}
    
    whisper_endpoint = "https://api.openai.com/v1/audio/translations"
    
    headers = {
        'Authorization': 'Bearer YOUR_OPENAI_API_KEY',
        'Content-Type': 'multipart/form-data'
    }
    
    response = requests.post(whisper_endpoint, headers=headers, files=files)
    
    if response.status_code == 200:
        transcription = response.json().get('text', '')
        return transcription
    else:
        return None

# Function to detect voice based on FFT
def is_speech_based_on_fft(data, sample_rate=16000, threshold=50):
    # Convert byte data to numpy array
    audio_signal = np.frombuffer(data, dtype=np.int16)
    
    # Apply FFT to the audio signal
    fft_spectrum = np.abs(fft(audio_signal))[:len(audio_signal) // 2]  # Take half of the spectrum
    freqs = np.fft.fftfreq(len(fft_spectrum) * 2, 1/sample_rate)[:len(fft_spectrum)]
    
    # Focus on the frequency range typical of human speech (300 Hz - 3000 Hz)
    lower_bound = np.where(freqs >= 300)[0][0]
    upper_bound = np.where(freqs <= 3000)[0][-1]
    
    speech_energy = np.mean(fft_spectrum[lower_bound:upper_bound])  # Mean energy in the speech range
    
    # Determine if the energy in this range exceeds the threshold
    return speech_energy > threshold

# Function to detect voice and handle buffer
def process_audio(data):
    global audio_buffer, is_talking, silence_counter
    
    is_speech_fft = is_speech_based_on_fft(data)  # Use FFT for detection
    is_speech_vad = vad.is_speech(data, sample_rate=16000)  # VAD check

    # Update the speaking status based on both FFT and VAD
    is_currently_talking = is_speech_fft and is_speech_vad
    
    if is_currently_talking and not is_talking:  # Just started talking
        print("Empecé a hablar.")
        is_talking = True
        silence_counter = 0  # Reset silence counter
    
    if is_currently_talking:
        audio_buffer.append(data)  # Accumulate audio while talking
        silence_counter = 0  # Reset silence counter during speech
    
    if not is_currently_talking and is_talking:
        # Increment the silence counter
        silence_counter += 1
    
        # If enough silent frames have passed, stop talking
        if silence_counter > SILENCE_FRAMES_THRESHOLD:  # Wait for N frames before confirming stop
            print("Dejé de hablar.")
            send_audio_to_whisper(audio_buffer)  # Send accumulated audio to Whisper
            audio_buffer = []  # Clear the buffer
            is_talking = False

# Global buffer to accumulate small audio chunks
partial_frame = b''

def audio_callback(indata, frames, time, status):
    global partial_frame

    if status:
        print(f"Sounddevice error: {status}")
        return

    # Convert to 16-bit PCM format
    audio_data = indata[:, 0].astype(np.int16).tobytes()

    # Append new audio to partial_frame buffer
    partial_frame += audio_data

    # Process audio in fixed-size chunks expected by VAD
    frame_duration_ms = 10  # We use 10ms frames
    frame_length = int(16000 * frame_duration_ms / 1000)  # 160 samples for 10ms at 16kHz

    # Process each frame (10ms at a time) from the accumulated buffer
    while len(partial_frame) >= frame_length * 2:
        frame = partial_frame[:frame_length * 2]
        partial_frame = partial_frame[frame_length * 2:]  # Remove processed frame from buffer
        process_audio(frame)

# Capture audio using sounddevice from the microphone
async def capture_audio():
    sample_rate = 16000
    channels = 1
    try:
        with sd.InputStream(callback=audio_callback, channels=channels, samplerate=sample_rate, dtype='int16'):
            await asyncio.sleep(60)  # Keep capturing audio for 60 seconds (adjust as needed)
    except Exception as e:
        print(f"Error capturing audio: {e}")

# Main with asyncio
if __name__ == "__main__":
    asyncio.run(capture_audio())
