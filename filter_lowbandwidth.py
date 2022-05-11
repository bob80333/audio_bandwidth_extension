import librosa
import numpy as np

from tqdm import tqdm
import argparse
from pathlib import Path
import soundfile
import torch
import torchaudio


# implement power spectrogram
def spec(audio):
    S = np.abs(librosa.stft(
        audio, n_fft=2400, hop_length=600, win_length=2400,
        window="hann", center=True, pad_mode="constant"
    )) ** 2
    return S


def rolloff_filtered(audio, db=50, t=0.99):
    S = spec(audio)
    S_log = np.log(S)
    max_db = np.max(S_log)
    db_filter = max_db - db
    S_log[S_log < db_filter] = -10000
    S = np.exp(S_log)
    total_energy = np.cumsum(S, axis=-2)
    threshold = t * total_energy[..., -1, :]
    threshold = np.expand_dims(threshold, axis=-2)
    ind = np.where(total_energy < threshold, np.nan, 1)
    freq = np.expand_dims(librosa.fft_frequencies(48000, 2400), axis=-1)
    return np.nanmin(ind * freq, axis=-2, keepdims=True)


def spec_torch(audio):
    S = torchaudio.functional.spectrogram(
        audio, n_fft=2400, hop_length=600, win_length=2400,
        window=torch.hann_window(2400).cuda(), center=True, pad_mode="constant", power=2, pad=0, normalized=False
    )
    return S


def torch_rolloff_filtered(audio, db=50, t=0.99):
    S = spec_torch(audio)
    S_log = torch.log(S)
    max_db = torch.max(S_log)
    db_filter = max_db - db
    S_log[S_log < db_filter] = -10000
    S = torch.exp(S_log)
    total_energy = torch.cumsum(S, dim=-2)
    threshold = t * total_energy[..., -1, :]
    threshold = torch.unsqueeze(threshold, dim=-2)
    ind = torch.where(total_energy < threshold, torch.tensor(np.nan, dtype=audio.dtype).cuda(), torch.tensor(1, dtype=audio.dtype).cuda())
    freq = torch.unsqueeze(torch.tensor(librosa.fft_frequencies(48000, 2400)).cuda(), dim=-1)
    freqs = torch.min(torch.nan_to_num(ind * freq), dim=-2, keepdim=True).values
    return freqs


def max_frequency(audio):
    return np.max(rolloff_filtered(audio))


# this will better find max frequency for audio with high frequency noise
def max_frequency_aggresive(audio):
    return np.max(rolloff_filtered(audio, db=20, t=0.95))


# the difference between regular and aggresive will determine false positives from regular


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("input_dir", type=str, help="input directory of wav files to process")
    args.add_argument("output_dir", type=str, help="output directory of wav files that are filtered to be high "
                                                   "bandwidth")

    args = args.parse_args()

    input_files = list(Path(args.input_dir).glob("*.wav"))

    total_allowed = 0

    for input_file in tqdm(input_files):
        audio, sr = librosa.load(str(input_file), sr=48000)
        max_freq = max_frequency(audio)
        aggressive = max_frequency_aggresive(audio)
        if max_freq < 22500 or aggressive / max_freq < 0.75:
            continue
        output_file = Path(args.output_dir) / input_file.name
        soundfile.write(str(output_file), audio, sr, subtype="PCM_16")
        total_allowed += 1

    print("Total allowed:", total_allowed)
    print("Total files:", len(list(input_files)))
    print("Ratio:", total_allowed / len(list(input_files)))
