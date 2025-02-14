from huggingface_hub import hf_hub_download
import torch
import os
import soundfile as sf
import random
import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.signal import resample_poly
from moshi.models import loaders, LMGen


# load the mimi and moshi models
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device='cuda')
mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi.
mimi.cuda()

moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
moshi = loaders.get_moshi_lm(moshi_weight, device='cuda')
lm_gen = LMGen(moshi, temp=0.8, temp_text=0.7)  # this handles sampling params etc.

@torch.no_grad()
def experiment(
    output_dir
):
    print("-----------------------------")
    print("output_path:", output_dir)
    print("-----------------------------")

    # load dataset
    with open("../advanced-voice-gen-task-v1/questions1_shuffled_id.json", "r") as f:
        tmp_dataset = json.load(f)
    print("len(dataset):", len(tmp_dataset))

    ids = [i for i in range(len(tmp_dataset))]
    # random.shuffle(ids)

    for i in tqdm(ids):
        output_file = f"{output_dir}/audio/{i}.wav"
        # check if the file already exists
        # if os.path.exists(output_file):
        #     print(f"Skipping {output_file} as it already exists.")
        #     continue

        # Load the input WAV file
        input_file = f"../advanced-voice-gen-task-v1/questions1_kokoro_wav/{i}.kokoro.wav"
        wav, sample_rate = sf.read(input_file)

        # Ensure the audio is mono
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)  # Convert stereo to mono

        # Resample if needed
        target_sample_rate = 24000
        if sample_rate != target_sample_rate:
            print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz...")
            wav = resample_poly(wav, target_sample_rate, sample_rate)

        zeros = np.zeros(mimi.sample_rate*5)  # Create an array of zeros
        wav = np.concatenate((zeros, wav))

        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cuda')

        # codes0 = mimi.encode(wav)  # [B, K = 8, T]
        # decoded = mimi.decode(codes)
        # print("codes0.shape:", codes0.shape)

        # Supports streaming too.
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        all_codes = []
        with mimi.streaming(batch_size=1):
            for offset in range(0, wav.shape[-1], frame_size):
                frame = wav[:, :, offset: offset + frame_size]
                if frame.shape[-1] != frame_size: # frame_size = 1920
                    break
                codes = mimi.encode(frame) # [B, K = 8, 1] -- I think
                assert codes.shape[-1] == 1, codes.shape
                all_codes.append(codes)

        ## WARNING: When streaming, make sure to always feed a total amount of audio that is a multiple
        #           of the frame size (1920), otherwise the last frame will not be complete, and thus
        #           will not be encoded. For simplicity, we recommend feeding in audio always in multiple
        #           of the frame size, so that you always know how many time steps you get back in `codes`.

        out_wav_chunks = []
        # Now we will stream over both Moshi I/O, and decode on the fly with Mimi.
        with lm_gen.streaming(1), mimi.streaming(1):
            for idx, code in enumerate(all_codes):
                tokens_out = lm_gen.step(code.cuda())
                # tokens_out is [B, 1 + 8, 1], with tokens_out[:, 1] representing the text token.
                if tokens_out is not None:
                    wav_chunk = mimi.decode(tokens_out[:, 1:])
                    out_wav_chunks.append(wav_chunk)
                print(idx, end='\r')
        out_wav = torch.cat(out_wav_chunks, dim=-1)
        out_wav_arr = out_wav.cpu().numpy().squeeze()
        sf.write(output_file, out_wav_arr, 24000)
        print(f"Saved to {output_file}")
        import ipdb; ipdb.set_trace()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    args = parser.parse_args()
    experiment(args.output_dir)

    # usage: python inference_advvoiceq1_moshi.py --output_dir experiments/advvoiceq1/moshi

if __name__ == "__main__":
    main()