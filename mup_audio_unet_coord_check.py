from mup import set_base_shapes, make_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data
# construct a dictionary of lazy μP models with differing widths
from torch.utils.data import DataLoader

from data import AudioDataset
from mup_audio_unet import VeryShallowMupAudioUNet

if __name__ == '__main__':
    # set the base shapes
    temp_model = VeryShallowMupAudioUNet(1)
    temp2_model = VeryShallowMupAudioUNet(2)
    base_shapes = make_base_shapes(temp_model, temp2_model, 'audiounet_base.bsh')

    def relevant_modules(name):
        if name in ['decoder_blocks.0.layers.0.layers.0', 'decoder_blocks.1.layers.0.layers.0', 'decoder_blocks.2.layers.0.layers.0', 'decoder_blocks.3.layers.0.layers.0',
                    'encoder_blocks.0.layers.0.layers.0', 'encoder_blocks.1.layers.0.layers.0', 'encoder_blocks.2.layers.0.layers.0', 'encoder_blocks.3.layers.0.layers.0']:
            return True
        return False

    def lazy_model(width):
        # `set_base_shapes` returns the model
        return lambda: set_base_shapes(VeryShallowMupAudioUNet(width), 'audiounet_base.bsh')
        # Note: any custom initialization with `mup.init` would need to
        # be done inside the lambda as well


    models = {2: lazy_model(2), 4: lazy_model(4), 8: lazy_model(8), 16: lazy_model(16), 32: lazy_model(32), 64: lazy_model(64), 128: lazy_model(128)}
    # make a dataloader with small batch size/seq len
    #   just for testing
    data = AudioDataset("D:/speech_enhancement/VCTK_noised/clean_trainset_56spk_wav", aug_prob=0,
                        test=False, segment_len=64000, dual_channel=False)
    dataloader = DataLoader(data, batch_size=16, shuffle=False, num_workers=2)
    # record data from the model activations over a few steps of training
    # this returns a pandas dataframe
    df = get_coord_data(models, dataloader, optimizer='adam', lr=3e-2, lossfn='l1')
    # This saves the coord check plots to filename.
    plot_coord_data(df, save_to="plot_mup_outkernel1_lastskip.png")

    def lazy_model_nomup(width):
        return lambda: VeryShallowMupAudioUNet(width, ismup=False)

    nomup_models = {2: lazy_model_nomup(2), 4: lazy_model_nomup(4), 8: lazy_model_nomup(8), 16: lazy_model_nomup(16), 32: lazy_model_nomup(32), 64: lazy_model_nomup(64), 128: lazy_model_nomup(128)}

    nomup_df = get_coord_data(nomup_models, dataloader, optimizer='adam', lr=3e-2, lossfn='l1', mup=False)
    # This saves the coord check plots to filename.
    plot_coord_data(nomup_df, save_to="plot_nomup_outkernel1_lastskip.png")
