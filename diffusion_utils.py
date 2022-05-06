# based on crowsonkb's diffusion code from v-diffusion-pytorch

import torch
import math
from tqdm.auto import trange

import torchaudio

from audio_unet_diffusion import get_model


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean audio and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def make_eps_model_fn(model):
    def eps_model_fn(x, t, **extra_args):
        alphas, sigmas = t_to_alpha_sigma(t)
        v = model(x, t, **extra_args)
        eps = x * sigmas[:, None, None] + v * alphas[:, None, None]
        return eps

    return eps_model_fn


# for some reason autocast makes my audio unet almost 2x slower....
# so disable it for now
def make_autocast_model_fn(model, enabled=False):
    def autocast_model_fn(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled):
            return model(*args, **kwargs).float()

    return autocast_model_fn


def transfer(x, eps, t_1, t_2):
    alphas, sigmas = t_to_alpha_sigma(t_1)
    next_alphas, next_sigmas = t_to_alpha_sigma(t_2)
    pred = (x - eps * sigmas[:, None, None]) / alphas[:, None, None]
    x = pred * next_alphas[:, None, None] + eps * next_sigmas[:, None, None]
    return x, pred


def prk_step(model, x, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    t_mid = (t_2 + t_1) / 2
    eps_1 = eps_model_fn(x, t_1, **extra_args)
    x_1, _ = transfer(x, eps_1, t_1, t_mid)
    eps_2 = eps_model_fn(x_1, t_mid, **extra_args)
    x_2, _ = transfer(x, eps_2, t_1, t_mid)
    eps_3 = eps_model_fn(x_2, t_mid, **extra_args)
    x_3, _ = transfer(x, eps_3, t_1, t_2)
    eps_4 = eps_model_fn(x_3, t_2, **extra_args)
    eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred


def plms_step(model, x, old_eps, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps = eps_model_fn(x, t_1, **extra_args)
    eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)
    return x_new, eps, pred


@torch.no_grad()
def plms_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth order
    Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in trange(len(steps) - 1, disable=None):
        if len(old_eps) < 3:
            x, eps, pred = prk_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        else:
            x, eps, pred = plms_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


def log_snr_to_alpha_sigma(log_snr):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    return log_snr.sigmoid().sqrt(), log_snr.neg().sigmoid().sqrt()


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


def get_ddpm_schedule(ddpm_t):
    """Returns timesteps for the noise schedule from the DDPM paper."""
    log_snr = -torch.special.expm1(1e-4 + 10 * ddpm_t ** 2).log()
    alpha, sigma = log_snr_to_alpha_sigma(log_snr)
    return alpha_sigma_to_t(alpha, sigma)


def get_spliced_ddpm_cosine_schedule(t):
    """Returns timesteps for a spliced DDPM/cosine noise schedule."""
    ddpm_crossover = 0.48536712
    cosine_crossover = 0.80074257
    big_t = t * (1 + cosine_crossover - ddpm_crossover)
    ddpm_part = get_ddpm_schedule(big_t + ddpm_crossover - cosine_crossover)
    return torch.where(big_t < cosine_crossover, big_t, ddpm_part)


if __name__ == "__main__":
    # how to sample:

    wav, sr = torchaudio.load("D:/speech_enhancement/VCTK_noised/clean_testset_wav/p232_002.wav")

    if wav.shape[-1] % 256 != 0:
        padding = 256 - wav.shape[-1] % 256
        wav = torch.cat([wav, torch.zeros((1, padding))], dim=-1)

    # resample audio to get low-quality audio

    bad = torchaudio.transforms.Resample(sr, 16000)(wav)
    bad_resampled = torchaudio.transforms.Resample(16000, sr)(bad)
    if bad_resampled.shape[-1] < wav.shape[-1]:
        padding = torch.zeros((1, wav.shape[-1] - bad_resampled.shape[-1]))
        bad_resampled = torch.cat((bad_resampled, padding), dim=-1)

    if bad_resampled.shape[-1] > wav.shape[-1]:
        bad_resampled = bad_resampled[:, :wav.shape[-1]]

    x = torch.randn(1, 1, wav.shape[-1])
    bad_resampled = bad_resampled.unsqueeze(0)

    model = get_model(width=2)
    model.eval()

    t = torch.linspace(1, 0, 50+1)[:-1]
    step_list = get_spliced_ddpm_cosine_schedule(t)

    denoised = plms_sample(model, x, step_list, {"condition_audio": bad_resampled}, is_reverse=False)
