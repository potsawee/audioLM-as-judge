import torch
import librosa
import glob
from tqdm import tqdm
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import pandas as pd
import base64
import io
import soundfile as sf
import pdb
from datasets import load_dataset

def get_encoded_audio_from_path(filepath):
    """
    Encodes an audio file to a Base64 string.

    Args:
        filepath (str): Path to the audio file.
    
    Returns:
        str: Base64-encoded string of the audio file.
    """
    with open(filepath, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def get_encoded_audio_from_array(audio_array, sample_rate):
    """
    Encodes a NumPy audio array to a Base64 string.
    
    Args:
        audio_array (numpy.ndarray): Audio data as a NumPy array.
        sample_rate (int): Sample rate of the audio.
    
    Returns:
        str: Base64-encoded string of the audio array.
    """
    # Save the audio array to a buffer in WAV format
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, samplerate=sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def load_and_preprocess_audio(audio_path, processor):
    """Load and preprocess the audio from the given path."""
    audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    return audio

def create_conversation(audio_path):
    """Create the conversation input for the model."""
    return [
        {"role": "system", "content": "You are an expert in audio quality assessment."},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": audio_path,
                },
                {"type": "text", "text": (
                    "You are an expert in audio quality assessment."
                    "Evaluate the quality of this audio and assign a score between 1.0 and 5.0 (inclusive). "
                    "Focus on clarity, fidelity, and overall listening experience."
                    "The output must be strictly only the final score in floating-point format, for example: [[floating point score]]."
                )}
            ]
        }
    ]
def create_ab_testing_conversation():
    """Create the conversation input for A/B testing with two audio files."""
    return [
        {"role": "system", "content": "You are an expert in audio quality assessment."},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "Audio A",
                },
                {
                    "type": "audio",
                    "audio_url": "Audio B",
                },
                {"type": "text", "text": (
                    "You are an expert in audio quality assessment. "
                    "Listen to both audio samples and determine which audio has higher quality. "
                    "The output must strictly be the audio with better quality, like: "
                    "[[Audio A]]"
                )}
            ]
        }
    ]

def qwen2_mos(audio_array, model, processor, device):
    """Evaluate the quality of the provided audio using the model."""
    # Load and preprocess the audio
    

    # Create the conversation input
    conversation = create_conversation(audio_array)

    # Prepare the input text and audio for the model
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio_array], return_tensors="pt", padding=True)

    # Move inputs to the device (GPU if available)
    inputs.input_ids = inputs.input_ids.to(device)
    inputs.attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        # Generate the model's response
        generate_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=256
        )

    # Decode and process the response
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response

def qwen2_ab_testing(audio_array_1, audio_array_2, model, processor, device):
    """Evaluate the quality of two audio samples using the model for A/B testing."""
    # Create the conversation input for A/B testing
    conversation = create_ab_testing_conversation()

    # Prepare the input text and audio for the model
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio_array_1, audio_array_2], return_tensors="pt", padding=True)

    # Move inputs to the device (GPU if available)
    inputs.input_ids = inputs.input_ids.to(device)
    inputs.attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        # Generate the model's response
        generate_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=256
        )

    # Decode and process the response
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response


def evaluate_better_audio(ds_row, model , processor, device):
    """
    Evaluates which audio in a dataset row has better quality.
    
    Args:
        ds_row (dict): A single row from the Hugging Face dataset containing
                       'path_chosen' and 'path_rejected' audio data.
    
    Returns:
        str: Decision on which audio is better ("[[Audio A]]" or "[[Audio B]]").
    """
    chosen_audio = ds_row['path_chosen']
    rejected_audio = ds_row['path_rejected']

    # Encode chosen and rejected audio to Base64
    chosen_array = (chosen_audio['array'])
    rejected_array = (rejected_audio['array'])
    # Perform A/B testing and return the result
    return qwen2_ab_testing(chosen_array, rejected_array, model , processor, device)

def main():
    """
    Main function to evaluate audio quality using GPT-4o.
    Loads a dataset and evaluates the better audio for a single example.
    """
    # Load the processor and model, specifying GPU usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"
    )
    # Load the dataset from Hugging Face
    ds = load_dataset("scb10x/tts_arena_resynthesized")
    audio_array = load_and_preprocess_audio('decoded_audio.wav', processor)
    result = evaluate_better_audio(ds['train'][1], model, processor, device)
    print(result)
    pdb.set_trace()
    result = qwen2_mos(audio_array, model, processor, device)
    print(result)


if __name__ == "__main__":
    main()
