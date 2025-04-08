import torch
import librosa
import glob
from tqdm import tqdm
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import pandas as pd
import base64
import io
import soundfile as sf
from datasets import load_dataset

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
                    "audio_path": audio_path,
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

def create_conversation_trs(audio_path):
    """Create the conversation input for the model."""
    return [
        {"role": "system", "content": "You are an expert in audio transcription."},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_path": audio_path,
                },
                {"type": "text", "text": "Transcribe this audio."}
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
                    "audio_path": "Audio A",
                },
                {
                    "type": "audio",
                    "audio_path": "Audio B",
                },
                {"type": "text", "text": (
                    "Listen to both audio samples and determine which audio has higher quality. "
                    "The output must strictly be the audio with better quality, like: [[Audio A]]."
                )}
            ]
        }
    ]

def qwen2_mos(audio_path, model, processor, device):
    """Evaluate the quality of the provided audio using the model."""
    # Load and preprocess the audio
    audio_array = load_and_preprocess_audio(audio_path, processor)

    # Create the conversation input
    conversation = create_conversation(audio_path)

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

def qwen2_ab_testing(audio_path_1, audio_path_2, model, processor, device):
    """Evaluate the quality of two audio samples using the model for A/B testing."""
    # Load and preprocess the audio
    audio_array_1 = load_and_preprocess_audio(audio_path_1, processor)
    audio_array_2 = load_and_preprocess_audio(audio_path_2, processor)

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


def qwen2_transcribe(audio_path, model, processor, device):
    """Evaluate the quality of the provided audio using the model."""
    # Load and preprocess the audio
    audio_array = load_and_preprocess_audio(audio_path, processor)

    # Create the conversation input
    conversation = create_conversation_trs(audio_path)

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


def main():
    """Main function to evaluate audio quality using Qwen2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"
    )

    # Example: Evaluate single audio file
    audio_path = "decoded_audio.wav"
    transcription = qwen2_transcribe(audio_path, model, processor, device)
    print(f"transcription: {transcription}")


    # Example: Evaluate single audio file
    audio_path = "decoded_audio.wav"
    mos_result = qwen2_mos(audio_path, model, processor, device)
    print(f"Mean Opinion Score (MOS): {mos_result}")

    # Example: A/B testing
    audio_path_1 = "decoded_audio.wav"
    audio_path_2 = "decoded_audio.wav"
    ab_result = qwen2_ab_testing(audio_path_1, audio_path_2, model, processor, device)
    print(f"A/B Testing Result: {ab_result}")

if __name__ == "__main__":
    main()
