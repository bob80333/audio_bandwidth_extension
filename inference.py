import os

from hybrid_unet import get_model
import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Hybrid Unet inference on an input directory of audio files, and save the output to an output directory.")

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory containing audio files to be processed.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to directory to save processed audio files.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model file.')

    args = parser.parse_args()

    model = get_model(device="cuda")

    model.load_state_dict(torch.load(
        args.model_path)['model'])

    model.eval()
    model.cuda()

    files = list(Path(args.input_dir).glob('*.wav'))

    with torch.inference_mode():
        for file in tqdm(files):
            # skip if output file already exists
            # can be used to resume processing if interrupted or crashed
            if os.path.exists(str(Path(args.output_dir) / file.name)):
                continue
            audio, sr = torchaudio.load(str(file))
            audio = audio.unsqueeze(0)

            audio_len = audio.shape[-1]

            pad_len = 0
            if audio_len % 256 != 0:
                pad_len = 256 - (audio_len % 256)
                shape = list(audio.shape)
                shape[-1] = pad_len
                padding = torch.zeros(shape)
                audio = torch.cat((audio, padding), dim=-1)

            audio = audio.cuda()

            output = model(audio)

            if pad_len != 0:
                output = output[:, :, :(output.shape[-1] - pad_len)]

            torchaudio.save(str(Path(args.output_dir) / file.name), output.cpu()[0], sr)
