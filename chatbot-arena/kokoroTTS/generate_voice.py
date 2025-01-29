import os
import argparse
import json
import random
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
from kokoro import KPipeline

def synthesize_speech(
    pipeline,
    text: str,
    voice_name: str,
    wav_path: str, # to save the audio file
):
    # check if wav_path already exists
    if os.path.exists(wav_path):
        print(f"Already exists: {wav_path}")
        return

    # Generate audio in chunks, but do not write them individually
    generator = pipeline(
        text, 
        voice=voice_name,
        speed=1, 
        split_pattern=r'\n+'
    )

    # Collect chunks of audio in a list
    audio_chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        # print(i)
        # print(gs)
        # print(ps)

        # Append this chunk to our collection
        audio_chunks.append(audio)

    # Combine all chunks into one NumPy array (at the original 24 kHz sample rate)
    combined_audio_24k = np.concatenate(audio_chunks)

    # ----- RESAMPLE from 24 kHz to 16 kHz -----
    # Librosa's resampling
    combined_audio_16k = librosa.resample(
        y=combined_audio_24k,
        orig_sr=24000,
        target_sr=16000
    )

    # Save the 16 kHz version to a single file
    sf.write(wav_path, combined_audio_16k, 16000)
    print(f"Saved to {wav_path}")

def experiment(
    input_path: str,
    wav_dir: str,
):
    print("-------------------------------------------")
    print("input_path:", input_path)
    print("wav_dir:", wav_dir)
    print("cuda:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("-------------------------------------------")

    pipelines = {
        'a': KPipeline(lang_code='a'), # american
        'b': KPipeline(lang_code='b'), # british
    }

    with open(input_path, "r") as f:
        data = json.load(f)
    print("len(data):", len(data))

    random.shuffle(data)

    for x in tqdm(data):
        # Get the analysis result
        assert x['conversation_a'][0]['content'] == x['conversation_b'][0]['content']
        # user turn
        user_turn_content = x['conversation_a'][0]['content']
        voice_user = x['voice_user']
        voice_user_wav_path = os.path.join(wav_dir, f"{x['question_id']}-user.wav")
        synthesize_speech(pipelines[voice_user[0]], user_turn_content, voice_user[1], voice_user_wav_path)

        # assistant turn a
        assistant_turn_a_content = x['conversation_a'][1]['content']
        voice_assistant_a = x['voice_a']
        voice_assistant_a_wav_path = os.path.join(wav_dir, f"{x['question_id']}-assistant-a.wav")
        synthesize_speech(pipelines[voice_assistant_a[0]], assistant_turn_a_content, voice_assistant_a[1], voice_assistant_a_wav_path)

        # assistant turn b
        assistant_turn_b_content = x['conversation_b'][1]['content']
        voice_assistant_b = x['voice_b']
        voice_assistant_b_wav_path = os.path.join(wav_dir, f"{x['question_id']}-assistant-b.wav")
        synthesize_speech(pipelines[voice_assistant_b[0]], assistant_turn_b_content, voice_assistant_b[1], voice_assistant_b_wav_path)

    print("Done.")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Analyze comparison between A and B from input text.")
    parser.add_argument("--input_path", 
        type=str, 
        default="/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/chatbot-arena-spoken-1turn-english-difference-voices.json",
        help="Path to the input file containing text to analyze."
    )
    parser.add_argument("--wav_dir", 
        type=str, 
        default="/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav",
        help="Path to the output file to save the synthesis results."
    )
    args = parser.parse_args()
    experiment(args.input_path, args.wav_dir)