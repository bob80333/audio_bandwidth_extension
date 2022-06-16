from mup import set_base_shapes

from mup_audio_unet import get_model
import argparse
import torch
import torchaudio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load an audio file, remove all frequencies above a threshold, then run audio unet inference on it to recover the frequencies, and save the output to an output directory.")

    parser.add_argument('--input', type=str, required=True,
                        help='Input file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file')
    parser.add_argument('--max_freq', type=int, default=4000)
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model file.')

    args = parser.parse_args()

    model = get_model(width=32)

    base_model = get_model(width=1)
    delta_model = get_model(width=2)

    model.load_state_dict(torch.load(
        args.model_path)['model'])

    set_base_shapes(model, base_model, delta=delta_model)

    model.eval()
    model.cuda()

    file = args.input

    with torch.inference_mode():
        audio, sr = torchaudio.load(str(file))
        audio = audio.unsqueeze(0)

        print(audio.shape)
        low_res_audio = torchaudio.transforms.Resample(sr, args.max_freq*2)(audio)
        print(low_res_audio.shape)
        low_quality_audio = torchaudio.transforms.Resample(args.max_freq*2, sr)(low_res_audio)
        print(low_quality_audio.shape)
        torchaudio.save(args.output.split('.wav')[0] + '_low_res.wav', low_quality_audio[0], sr)
        audio = low_quality_audio

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

        output = output + audio

        if pad_len != 0:
            output = output[:, :, :(output.shape[-1] - pad_len)]

        torchaudio.save(args.output, output.cpu()[0], sr)
