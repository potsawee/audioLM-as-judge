import os
import argparse
import random
import io
import pydub
from tqdm import tqdm
from datasets import load_from_disk
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

# Retrieve the ElevenLabs API key from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Raise an error if the API key is not set
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

# Initialize the ElevenLabs client with the API key
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def text_to_speech_file(
    wav_save_path,
    text, 
    voice_id, 
    # style_level=0
):
    """
    Converts text to speech and saves the output as an MP3 file.

    Args:
        text (str): The text content to convert to speech.
        voice_id (str): The ID of the voice to use for the speech.
        style_level (int): The style level for the voice settings.

    Returns:
        str: The file path where the audio file has been saved.
    """
    # Call the text-to-speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id=voice_id,  # Voice ID to use for the speech
        optimize_streaming_latency="0",  # Optimize for streaming latency
        output_format="mp3_44100_128",  # generate MP3 (44.1 kHz, 128 kbps)
        text=text,  # Text content to convert to speech
        model_id="eleven_multilingual_v2",  # Model ID for the conversion
        voice_settings=VoiceSettings(
            stability=0.0,  # Stability setting for the voice
            similarity_boost=1.0,  # Similarity boost setting for the voice
            # style=style_level,  # Style level for the voice
            use_speaker_boost=True,  # Use speaker boost setting
        ),
    )
    # Write response chunks to an in-memory buffer
    audio_data = io.BytesIO()
    for chunk in response:
        audio_data.write(chunk)

    audio_data.seek(0)
    # Decode MP3 to pydub's AudioSegment
    audio_segment = pydub.AudioSegment.from_file(audio_data, format="mp3")
    # Resample to 16 kHz (and optionally make it mono if you want)
    resampled_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    resampled_segment.export(wav_save_path, format="wav")
    print(f"Saved 16 kHz WAV file to: {wav_save_path}")


def main(
    output_dir
):
    input_path = "./data-chatbot-arena-spoken-style-11labs"
    dataset = load_from_disk(input_path)

    ids = [i for i in range(len(dataset))]
    random.shuffle(ids)

    for i in tqdm(ids):
        x = dataset[i]
        coversation_id = x["id"]
        winner_style = x["winner_style"]
        wav_assistant_path = f"{output_dir}/{coversation_id}_{winner_style}.11labs.wav"
        # check if the wav file already exists
        if os.path.exists(wav_assistant_path):
            print(f"Skipping {wav_assistant_path}")
            continue

        if winner_style == "model_a":
            assistant_text = x["assistant_a"]
        elif winner_style == "model_b":
            assistant_text = x["assistant_b"]
        else:
            raise ValueError("Invalid winner style")

        text_to_speech_file(
            wav_save_path=wav_assistant_path, 
            text=assistant_text,
            voice_id=x['style']['voice_id'],
        )


    print("finish synthetizing")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()
    main(args.output_dir)

    # usage: python tts_elevenLabs_generate_chatbot_arena.py --output_dir generated_11labs_audio
    # elevenlabs.core.api_error.ApiError: status_code: 400, body: {'detail': {'status': 'voice_not_found', 'message': 'A voice for the voice_id 1TE7ou3jyxHsyRehUuMB was not found.'}}
    # if it shows this error, please go to elevenLabs and add the voice to your account