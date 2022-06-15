import torch
from torch.utils.data import DataLoader
from data import AudioDataset
from mup_audio_unet import get_model
from asteroid import losses
import numpy as np
import torchaudio
import torch_ema
from tqdm import tqdm
import wandb

from mup import MuAdamW, set_base_shapes

import argparse

N_TRAIN_STEPS = 100_000
BATCH_SIZE = 16
ACCUMULATE_N = 2
N_SAMPLES_BASE = 64_000
EVAL_EVERY = 5000
START_EMA = 2_000
STEP = 1
SEGMENT_LEN_MULTIPLIER = 1
# optim hyperparams
LEARNING_RATE = 0.01
BETA_1 = 0
BETA_2 = 0
WEIGHT_DECAY = 0.1

# lr scheduler
LR_DECAY_GAMMA = 0.99999

# other hyperparams
EMA_DECAY = 0.999
CLIP_GRAD_NORM = 2.0

USE_AMP = False
# model parameters
WIDTH = 32
N_RES_UNITS = 3


NORM_TYPE = "no_norm"


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
    parser.add_argument('--ema_decay', type=float, default=EMA_DECAY)
    parser.add_argument('--clip_grad_norm', type=float, default=CLIP_GRAD_NORM)
    parser.add_argument('--prefix', type=str, default="test_")
    parser.add_argument("--beta_1", type=float, default=BETA_1)
    parser.add_argument("--beta_2", type=float, default=BETA_2)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--gamma", type=float, default=LR_DECAY_GAMMA)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    N_TRAIN_STEPS = args.n_train_steps
    ACCUMULATE_N = args.accumulate_n
    EVAL_EVERY = args.eval_every
    START_EMA = args.start_ema
    STEP = args.step
    LEARNING_RATE = args.learning_rate
    USE_AMP = args.use_amp
    SEGMENT_LEN_MULTIPLIER = args.segment_len_multiplier
    WIDTH = args.width
    N_RES_UNITS = args.n_res_units
    EMA_DECAY = args.ema_decay
    CLIP_GRAD_NORM = args.clip_grad_norm
    BETA_1 = args.beta_1
    BETA_2 = args.beta_2
    WEIGHT_DECAY = args.weight_decay
    LR_DECAY_GAMMA = args.gamma

    wandb.init(project="audio-bandwidth-extension", entity="bob80333")

    base_model = get_model(width=1, n_res_units=N_RES_UNITS)
    delta_model = get_model(width=2, n_res_units=N_RES_UNITS)

    model = get_model(width=WIDTH, n_res_units=N_RES_UNITS)

    set_base_shapes(model, base_model, delta=delta_model)

    model = model.cuda()

    # clean, then noisy
    # this will be the order the dataloader returns the audio in
    train_data = AudioDataset("D:/speech_enhancement/VCTK_noised/clean_trainset_56spk_wav", aug_prob=0,
                              test=False, segment_len=N_SAMPLES_BASE * SEGMENT_LEN_MULTIPLIER, dual_channel=False)

    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)

    eval_data = AudioDataset("D:/speech_enhancement/VCTK_noised/clean_testset_wav",
                             segment_len=48000 * 10, test=True, dual_channel=False)

    eval_dataloader = DataLoader(eval_data, batch_size=BATCH_SIZE, num_workers=3)

    optimizer = MuAdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, LR_DECAY_GAMMA)

    wandb.config.update({
        "learning_rate": LEARNING_RATE,
        "beta_1": BETA_1,
        "beta_2": BETA_2,
        "weight_decay": WEIGHT_DECAY,
        "lr_decay": "exponential",
        "gamma": LR_DECAY_GAMMA,
        "ema_decay": EMA_DECAY,
        "clip_grad_norm": CLIP_GRAD_NORM,
        "batch_size": BATCH_SIZE,
        "accumulate_n": ACCUMULATE_N,
        "train_segment_len": N_SAMPLES_BASE * SEGMENT_LEN_MULTIPLIER,
        "eval_segment_len": 48000 * 10,
        "n_train_steps": N_TRAIN_STEPS,
        "model_width": WIDTH,
        "step_size": STEP,
        "start_ema": START_EMA,
        "eval_every": EVAL_EVERY,
        "segment_len_multiplier": SEGMENT_LEN_MULTIPLIER,
        "model_type": "audio_unet",
        "prefix": args.prefix,
        "n_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "use_amp": USE_AMP,
        "predicted": "difference",
        "normalization_layer": NORM_TYPE,
    })

    loss_fn = losses.multi_scale_spectral.SingleSrcMultiScaleSpectral()
    loss_fn = loss_fn.cuda()

    sisdr_fn = losses.sdr.PairwiseNegSDR(sdr_type='sisdr')

    train_dataloader = infinite_dataloader(dataloader)

    best_sisdr = -100
    best_sisdr_ema = -100

    ema_model = torch_ema.ExponentialMovingAverage(model.parameters(), decay=EMA_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for i in tqdm(range(0, N_TRAIN_STEPS + 1, STEP)):
        loss_val = 0
        for j in range(ACCUMULATE_N):
            batch = next(train_dataloader)

            clean, degraded = batch
            clean = clean.cuda()
            degraded = degraded.cuda()

            difference = clean - degraded

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                estimated_difference = model(degraded)

                loss = loss_fn(estimated_difference, difference).mean()
                loss /= ACCUMULATE_N
                loss /= SEGMENT_LEN_MULTIPLIER
            loss_val += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

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
                sisdr_losses = []
                val_losses = []
                for batch in tqdm(eval_dataloader):
                    clean, degraded, start_idx, end_idx = batch
                    clean = clean.cuda()
                    degraded = degraded.cuda()

                    estimated_difference = model(degraded)
                    estimated_clean = estimated_difference + degraded

                    for est_clean, real_clean, start, end in zip(estimated_clean, clean, start_idx, end_idx):
                        est_clean = est_clean[:, start:end].unsqueeze(0)
                        real_clean = real_clean[:, start:end].unsqueeze(0)
                        sisdr_loss = -sisdr_fn(est_clean, real_clean)
                        val_loss = loss_fn(est_clean, real_clean).mean()
                        sisdr_losses.append(sisdr_loss.squeeze().item())
                        val_losses.append(val_loss.item())

                print("SI-SDR", np.mean(sisdr_losses))
                wandb.log({"si-sdr": np.mean(sisdr_losses)}, step=i)
                wandb.log({"val_loss": np.mean(val_losses)}, step=i)

                if np.mean(sisdr_losses) > best_sisdr:
                    best_sisdr = np.mean(sisdr_losses)
                    torch.save({"model": model.state_dict(), "si-sdr": best_sisdr},
                               args.prefix + NORM_TYPE + "_best_audio_unet_model_bandwidth_extension.pt")

                torchaudio.save(args.prefix + NORM_TYPE + "_sample_bandwith_extended_{}.wav".format(i), est_clean[0].cpu(),
                                48000)

                sisdr_losses_ema = []
                val_losses_ema = []
                with ema_model.average_parameters():
                    for batch in tqdm(eval_dataloader):
                        clean, degraded, start_idx, end_idx = batch
                        clean = clean.cuda()
                        degraded = degraded.cuda()

                        estimated_difference = model(degraded)
                        estimated_clean = estimated_difference + degraded

                        for est_clean, real_clean, start, end in zip(estimated_clean, clean, start_idx, end_idx):
                            est_clean = est_clean[:, start:end].unsqueeze(0)
                            real_clean = real_clean[:, start:end].unsqueeze(0)
                            sisdr_loss = -sisdr_fn(est_clean, real_clean)
                            val_loss = loss_fn(est_clean, real_clean).mean()
                            sisdr_losses_ema.append(sisdr_loss.squeeze().item())
                            val_losses_ema.append(val_loss.item())

                print("SI-SDR EMA", np.mean(sisdr_losses_ema))
                wandb.log({"si-sdr ema": np.mean(sisdr_losses_ema)}, step=i)
                wandb.log({"val_loss ema": np.mean(val_losses_ema)}, step=i)

                if np.mean(sisdr_losses_ema) > best_sisdr_ema:
                    best_sisdr_ema = np.mean(sisdr_losses_ema)
                    torch.save({"model": model.state_dict(), "si-sdr": best_sisdr_ema},
                               args.prefix + NORM_TYPE + "_best_audio_unet_ema_bandwidth_extension.pt")

                if i == 0:
                    torchaudio.save(args.prefix + NORM_TYPE + "_sample_degraded.wav".format(i), degraded[-1][:, start:end].cpu(),
                                    48000)
                    torchaudio.save(args.prefix + NORM_TYPE + "_sample_clean.wav".format(i), clean[-1][:, start:end].cpu(), 48000)
                torchaudio.save(args.prefix + NORM_TYPE + "_sample_bandwidth_extended_ema_{}.wav".format(i), est_clean[0].cpu(),
                                48000)

        if i == START_EMA:
            # restart EMA
            ema_model = torch_ema.ExponentialMovingAverage(model.parameters(), decay=EMA_DECAY)
