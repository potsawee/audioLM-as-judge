import os
import uuid
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

# Load environment variables from a .env file
load_dotenv()

# Retrieve the ElevenLabs API key from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Raise an error if the API key is not set
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

# Initialize the ElevenLabs client with the API key
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def text_to_speech_file(text, voice_id, style_level=0):
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
        output_format="mp3_44100_128",  # Output format for the audio file
        text=text,  # Text content to convert to speech
        model_id="eleven_multilingual_v2",  # Model ID for the conversion
        voice_settings=VoiceSettings(
            stability=0.0,  # Stability setting for the voice
            similarity_boost=1.0,  # Similarity boost setting for the voice
            style=style_level,  # Style level for the voice
            use_speaker_boost=True,  # Use speaker boost setting
        ),
    )

    # Generate a unique file name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Write the audio stream to the file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"A new audio file was saved successfully at {save_file_path}")

    # Return the path of the saved audio file
    return save_file_path

if __name__ == "__main__":
    # Example input sentence to convert to speech
    input_sentence = 'Not much, just trying to avoid this endless stream of small talk here.'
    # Call the text_to_speech_file function with the input sentence, voice ID, and the style level (from 0.0 to 1.0)
    text_to_speech_file(input_sentence, "cwP4aDXy5WMNpSu9jjek", 1.0)
