import base64
import io
from pydub import AudioSegment
import numpy as np

def encode_audio_array_with_resampling(audio_array: np.ndarray, 
                                       original_sr: int, 
                                       target_sample_rate: int = 16000) -> str:
    """
    Encode audio (provided as a NumPy array of float64) as base64 WAV, resampling if necessary.
    
    :param audio_array: NumPy array of audio samples. 
                       (float64, typically in [-1, 1])
    :param original_sr: The original sampling rate of the audio.
    :param target_sample_rate: Desired sampling rate for the output.
    :return: Base64-encoded string representing the WAV audio, or None if error.
    """
    try:
        # Ensure data is in [-1, 1] to avoid int16 overflow
        audio_array = np.clip(audio_array, -1.0, 1.0)

        # Convert float in [-1, 1] to 16-bit int. For example: 0.5 -> 16383
        audio_int16 = (audio_array * 32767.0).astype(np.int16)
        
        # Create an AudioSegment from the raw int16 data
        # Assuming the audio is mono; if stereo, set channels=2
        audio_segment = AudioSegment(
            data=audio_int16.tobytes(),
            sample_width=2,      # 16 bits = 2 bytes
            frame_rate=original_sr,
            channels=1
        )

        # Resample if needed
        if original_sr != target_sample_rate:
            print(f"Resampling from {original_sr} Hz to {target_sample_rate} Hz")
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)

        # Export to WAV in-memory
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()

        # Encode to Base64
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        return encoded_audio

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def encode_audio_with_resampling(filepath, target_sample_rate=16000):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(filepath)
        
        # Check the current sample rate
        if audio.frame_rate != target_sample_rate:
            print(f"Resampling from {audio.frame_rate} Hz to {target_sample_rate} Hz")
            audio = audio.set_frame_rate(target_sample_rate)
        
        # Export the audio to a temporary file or bytes
        audio_bytes = audio.export(format="wav").read()
        
        # Encode as Base64
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None