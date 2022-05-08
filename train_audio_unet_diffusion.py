import torch
from torch.utils.data import DataLoader
from data import AudioDataset
from audio_unet_diffusion import get_model
from asteroid import losses
import numpy as np
import torchaudio
import torch_ema
from tqdm import tqdm
import wandb

import argparse

from diffusion_utils import get_spliced_ddpm_cosine_schedule, t_to_alpha_sigma, plms_sample

N_TRAIN_STEPS = 50_000
BATCH_SIZE = 16
ACCUMULATE_N = 4
EVAL_EVERY = 2000
START_EMA = 2_000
STEP = 1
SEGMENT_LEN_MULTIPLIER = 1
LEARNING_RATE = 1e-3
USE_AMP = False
# test different number of diffusion steps
SAMPLING_STEPS = [8, 20, 50]
# model parameters
WIDTH = 32
N_RES_UNITS = 3
COND_WIDTH = 256

def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--n_train_steps', type=int, default=N_TRAIN_STEPS)
    parser.add_argument('--accumulate_n', type=int, default=ACCUMULATE_N)
    parser.add_argument('--eval_every', type=int, default=EVAL_EVERY)
    parser.add_argument('--start_ema', type=int, default=START_EMA)
    parser.add_argument('--step', type=int, default=STEP)
    parser.add_argument('--segment_len_multiplier', type=int, default=SEGMENT_LEN_MULTIPLIER)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--use_amp', type=bool, default=USE_AMP)
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--n_res_units', type=int, default=N_RES_UNITS)
    parser.add_argument('--cond_width', type=int, default=COND_WIDTH)
    parser.add_argument('prefix', type=str)

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    N_TRAIN_STEPS = args.n_train_steps
    ACCUMULATE_N = args.accumulate_n
    EVAL_EVERY = args.eval_every
    START_EMA = args.start_ema
    STEP = args.step
    LEARNING_RATE = args.learning_rate
    USE_AMP = args.use_amp
    WIDTH = args.width
    N_RES_UNITS = args.n_res_units
    COND_WIDTH = args.cond_width

    wandb.init(project="audio-bandwidth-extension", entity="bob80333")

    model = get_model(width=WIDTH, n_res_units=N_RES_UNITS, cond_width=COND_WIDTH)
    model = model.cuda()

    # clean, then noisy
    # this will be the order the dataloader returns the audio in
    train_data = AudioDataset("D:/speech_enhancement/VCTK_noised/clean_trainset_56spk_wav", aug_prob=0,
                              test=False, segment_len=48000 * 2 * SEGMENT_LEN_MULTIPLIER, dual_channel=False)

    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    eval_data = AudioDataset("D:/speech_enhancement/VCTK_noised/clean_testset_wav_small",
                             segment_len=48000 * 10, test=True, dual_channel=False)

    eval_dataloader = DataLoader(eval_data, batch_size=BATCH_SIZE, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9998)

    wandb.config.update({
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "accumulate_n": ACCUMULATE_N,
        "train_segment_len": 48000 * 2 * SEGMENT_LEN_MULTIPLIER,
        "eval_segment_len": 48000 * 10,
        "n_train_steps": N_TRAIN_STEPS,
        "step_size": STEP,
        "start_ema": START_EMA,
        "eval_every": EVAL_EVERY,
        "segment_len_multiplier": SEGMENT_LEN_MULTIPLIER,
        "model_type": "audio_unet_diffusion",
        "prefix": args.prefix,
        "n_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "lr_decay": "exponential",
        "gamma": 0.9998,
        "ema_decay": 0.999,
        "use_amp": USE_AMP,
        "diffusion_sampling_steps": SAMPLING_STEPS,
        "model_width": WIDTH,
        "n_res_units": N_RES_UNITS,
        "cond_width": COND_WIDTH,
    })

    loss_fn = losses.multi_scale_spectral.SingleSrcMultiScaleSpectral()
    loss_fn = loss_fn.cuda()

    sisdr_fn = losses.sdr.PairwiseNegSDR(sdr_type='sisdr')

    train_dataloader = infinite_dataloader(dataloader)

    best_sisdr = -100
    best_sisdr_ema = -100

    ema_model = torch_ema.ExponentialMovingAverage(model.parameters(), decay=0.999)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    step_list = []
    for SAMPLE_STEPS in SAMPLING_STEPS:
        ts = torch.linspace(1, 0, SAMPLE_STEPS + 1, ).cuda()[:-1]
        step_list.append(get_spliced_ddpm_cosine_schedule(ts))

    for i in tqdm(range(0, N_TRAIN_STEPS + 1, STEP)):
        loss_val = 0
        for j in range(ACCUMULATE_N):
            batch = next(train_dataloader)

            clean, degraded = batch
            clean = clean.cuda()
            degraded = degraded.cuda()

            # diffusion math:
            t = torch.rand(clean.shape[0]).cuda()
            step = get_spliced_ddpm_cosine_schedule(t)
            alphas, sigmas = t_to_alpha_sigma(step)
            noise = torch.randn(degraded.shape).cuda()
            # v-objective diffusion
            v = noise * alphas[:, None, None] - clean * sigmas[:, None, None]
            z = clean * alphas[:, None, None] + noise * sigmas[:, None, None]

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                estimated_v = model(z, timestep=step, condition_audio=degraded)

                loss = loss_fn(estimated_v, v).mean()
            loss_val += loss.item()
            scaler.scale(loss).backward()

        loss_val /= ACCUMULATE_N
        loss_val /= SEGMENT_LEN_MULTIPLIER
        scaler.step(optimizer)
        optimizer.zero_grad(set_to_none=True)

        if i % 8 == 0:
            wandb.log({"loss": loss_val, "lr": scheduler.get_last_lr()[0]}, step=i)

        # step scheduler after every step, and after logging lr
        scheduler.step()
        # step ema after every step
        ema_model.update()
        # step grad scaler
        scaler.update()

        # evaluate every EVAL_EVERY steps
        if i % EVAL_EVERY == 0:
            # torch.inference_mode() should be slightly faster than torch.no_grad()
            with torch.inference_mode():

                for SAMPLE_STEPS, steps in zip(SAMPLING_STEPS, step_list):
                    sisdr_losses = []
                    val_losses = []
                    for batch in tqdm(eval_dataloader):
                        clean, degraded, start_idx, end_idx = batch
                        clean = clean.cuda()
                        degraded = degraded.cuda()

                        noise = torch.randn(degraded.shape).cuda()
                        estimated_clean = plms_sample(model, noise, steps, {"condition_audio": degraded})

                        for est_clean, real_clean, start, end in zip(estimated_clean, clean, start_idx, end_idx):
                            est_clean = est_clean[:, start:end].unsqueeze(0)
                            real_clean = real_clean[:, start:end].unsqueeze(0)
                            sisdr_loss = -sisdr_fn(est_clean, real_clean)
                            val_loss = loss_fn(est_clean, real_clean).mean()
                            sisdr_losses.append(sisdr_loss.squeeze().item())
                            val_losses.append(val_loss.item())

                    print("SI-SDR " + str(SAMPLE_STEPS), np.mean(sisdr_losses))
                    wandb.log({"si-sdr " + str(SAMPLE_STEPS): np.mean(sisdr_losses)}, step=i)
                    wandb.log({"val_loss " + str(SAMPLE_STEPS): np.mean(val_losses)}, step=i)

                    if np.mean(sisdr_losses) > best_sisdr:
                        best_sisdr = np.mean(sisdr_losses)
                        torch.save({"model": model.state_dict(), "si-sdr": best_sisdr},
                                   str(SAMPLE_STEPS) + args.prefix + "best_diffusion_audio_unet_model_bandwidth_extension.pt")

                    torchaudio.save(str(SAMPLE_STEPS) + args.prefix + "diffusion_sample_bandwith_extended_{}.wav".format(i), est_clean[0].cpu(),
                                    48000)

                    sisdr_losses_ema = []
                    val_losses_ema = []
                    with ema_model.average_parameters():
                        for batch in tqdm(eval_dataloader):
                            clean, degraded, start_idx, end_idx = batch
                            clean = clean.cuda()
                            degraded = degraded.cuda()

                            noise = torch.randn(degraded.shape).cuda()
                            estimated_clean = plms_sample(model, noise, steps, {"condition_audio": degraded})

                            for est_clean, real_clean, start, end in zip(estimated_clean, clean, start_idx, end_idx):
                                est_clean = est_clean[:, start:end].unsqueeze(0)
                                real_clean = real_clean[:, start:end].unsqueeze(0)
                                sisdr_loss = -sisdr_fn(est_clean, real_clean)
                                val_loss = loss_fn(est_clean, real_clean).mean()
                                sisdr_losses_ema.append(sisdr_loss.squeeze().item())
                                val_losses_ema.append(val_loss.item())

                    print("SI-SDR EMA " + str(SAMPLE_STEPS), np.mean(sisdr_losses_ema))
                    wandb.log({"si-sdr ema " + str(SAMPLE_STEPS): np.mean(sisdr_losses_ema)}, step=i)
                    wandb.log({"val_loss ema " + str(SAMPLE_STEPS): np.mean(val_losses_ema)}, step=i)

                    if np.mean(sisdr_losses_ema) > best_sisdr_ema:
                        best_sisdr_ema = np.mean(sisdr_losses_ema)
                        torch.save({"model": model.state_dict(), "si-sdr": best_sisdr_ema},
                                   str(SAMPLE_STEPS) +args.prefix + "best_diffusion_audio_unet_ema_bandwidth_extension.pt")

                    if i == 0:
                        torchaudio.save(args.prefix + "sample_degraded.wav".format(i), degraded[-1][:, start:end].cpu(),
                                        48000)
                        torchaudio.save(args.prefix + "sample_clean.wav".format(i), clean[-1][:, start:end].cpu(), 48000)
                    torchaudio.save(str(SAMPLE_STEPS) + args.prefix + "diffusion_sample_bandwidth_extended_ema_{}.wav".format(i), est_clean[0].cpu(),
                                    48000)

        if i == START_EMA:
            # restart EMA
            ema_model = torch_ema.ExponentialMovingAverage(model.parameters(), decay=0.999)
