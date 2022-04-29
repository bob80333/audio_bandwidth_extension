from torch.utils.data import Dataset
import torchaudio
import torch

import numpy as np
import librosa

from pathlib import Path

AUDIO_SAMPLE_RATE = 48000
# have 48000 in the reduced sample rate list, so that the model will learn to pass through high quality audio
# without losing information
REDUCED_SAMPLE_RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000]


class AudioDataset(Dataset):

    def __init__(self, folder, segment_len=256 * 256, file_ext=".wav", test=False, aug_prob=0.25,
                 dual_channel=True):

        self.segment_len = segment_len

        self.files = list(Path(folder).rglob("*" + file_ext))

        self.test = test
        self.aug_prob = aug_prob
        self.dual_channel = dual_channel

        self.downsamplers = [torchaudio.transforms.Resample(AUDIO_SAMPLE_RATE, sr) for sr in REDUCED_SAMPLE_RATES]
        self.upsamplers = [torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE) for sr in REDUCED_SAMPLE_RATES]

        self.sampler_pairs = [(down, up) for down, up in zip(self.downsamplers, self.upsamplers)]

    def __len__(self):
        # pretend each different segment is a different sample to allow multiprocessed shuffled sampling easily
        return len(self.files)

    # assumes audio files are same length
    def __getitem__(self, item):
        # audio is shape (channels, samples)
        audio1, sr = torchaudio.load(str(self.files[item]))

        orig_length = audio1.shape[-1]

        if self.test:
            # test set has consistent sampler pairs for each item
            # as long as the audio is at the same index in the list
            current_sampler_pair = item % len(self.sampler_pairs)
            down, up = self.sampler_pairs[current_sampler_pair]
            audio2 = up(down(audio1))

        else:
            # randomly choose level of bandwidth degradation
            down, up = self.sampler_pairs[np.random.randint(len(self.sampler_pairs))]
            audio2 = up(down(audio1))

            # only apply augmentations some percent of the time
            if np.random.rand() < self.aug_prob:
                # apply augmentations
                speed_shift = np.random.choice(np.linspace(0.94, 1.06, num=25))
                pitch_shift = np.random.choice(np.linspace(-1, 1, num=3))

                audio1 = torch.tensor(librosa.effects.time_stretch(audio1.squeeze().numpy(), speed_shift)).unsqueeze(0)
                audio2 = torch.tensor(librosa.effects.time_stretch(audio2.squeeze().numpy(), speed_shift)).unsqueeze(0)

                audio1 = torch.tensor(librosa.effects.pitch_shift(audio1.squeeze().numpy(), sr, pitch_shift)).unsqueeze(
                    0)
                audio2 = torch.tensor(librosa.effects.pitch_shift(audio2.squeeze().numpy(), sr, pitch_shift)).unsqueeze(
                    0)

        # make sure audios are the same length after resampling
        if audio1.shape[-1] != audio2.shape[-1]:
            if audio1.shape[-1] > audio2.shape[-1]:
                # pad shorter audio2 with zeros
                padding = audio1.shape[-1] - audio2.shape[-1]
                audio2 = torch.cat((audio2, torch.zeros(1, padding)), dim=-1)
            else:
                # pad shorter audio1 with zeros
                padding = audio2.shape[-1] - audio1.shape[-1]
                audio1 = torch.cat((audio1, torch.zeros(1, padding)), dim=-1)

        trim_start = 0
        trim_end = orig_length

        if audio1.shape[1] > self.segment_len:
            end = audio1.shape[1] - self.segment_len
            start_idx = torch.randint(0, end, (1,)).item()

            audio1 = audio1[:, start_idx:start_idx + self.segment_len]
            audio2 = audio2[:, start_idx:start_idx + self.segment_len]

        elif audio1.shape[1] < self.segment_len:
            pad_len = self.segment_len - audio1.shape[-1]
            if pad_len % 2 == 0:
                padding = torch.zeros((1, pad_len // 2))
                audio1 = torch.cat((padding, audio1, padding), dim=1)
                audio2 = torch.cat((padding, audio2, padding), dim=1)
                trim_start = padding.shape[-1]
                trim_end = orig_length + padding.shape[-1]
            else:
                pad_1 = torch.zeros((1, 1))
                padding = torch.zeros((1, (pad_len - 1) // 2))
                audio1 = torch.cat((padding, pad_1, audio1, padding), dim=1)
                audio2 = torch.cat((padding, pad_1, audio2, padding), dim=1)
                trim_start = padding.shape[-1] + pad_1.shape[-1]
                trim_end = orig_length + pad_1.shape[-1] + padding.shape[-1]

        if self.dual_channel:
            audio1 = audio1.repeat(2, 1)
            audio2 = audio2.repeat(2, 1)

        assert audio1.shape == audio2.shape, "audio1 and audio2 should be the same shape, audio1:" + str(audio1.shape) + " audio2:" + str(audio2.shape)
        assert audio1.shape[-1] == self.segment_len, "audio should have segment length, audio1:" + str(audio1.shape[-1]) + " audio2:" + str(audio2.shape[-1])

        # audio 1 is the original high quality audio, audio 2 is the degraded audio
        # allow testing to clip audio back to original length
        if self.test:
            return audio1, audio2, trim_start, trim_end
        # return pair
        return audio1, audio2
