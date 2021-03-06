from mup import set_base_shapes, make_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data
# construct a dictionary of lazy μP models with differing widths
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data import AudioDataset
from mup_audio_unet import VeryShallowMupAudioUNet

if __name__ == '__main__':
    # set the base shapes
    temp_model = VeryShallowMupAudioUNet(1)
    temp2_model = VeryShallowMupAudioUNet(2)
    base_shapes = make_base_shapes(temp_model, temp2_model, 'audiounet_wn_base.bsh')

    def relevant_modules(name):
        if name in ['decoder_blocks.0.layers.0.layers.0', 'decoder_blocks.1.layers.0.layers.0', 'decoder_blocks.2.layers.0.layers.0', 'decoder_blocks.3.layers.0.layers.0',
                    'encoder_blocks.0.layers.0.layers.0', 'encoder_blocks.1.layers.0.layers.0', 'encoder_blocks.2.layers.0.layers.0', 'encoder_blocks.3.layers.0.layers.0']:
            return True
        return False

    def lazy_model(width):
        # `set_base_shapes` returns the model
        def get_model(width):
            model = VeryShallowMupAudioUNet(width, ismup=True)
            set_base_shapes(model, 'audiounet_wn_base.bsh')
            return model
        return lambda: get_model(width)
        # Note: any custom initialization with `mup.init` would need to
        # be done inside the lambda as well


    models = {2: lazy_model(2), 4: lazy_model(4), 8: lazy_model(8), 16: lazy_model(16), 32: lazy_model(32), 64: lazy_model(64), 128: lazy_model(128)}
    # make a dataloader with small batch size/seq len
    #   just for testing
    data = AudioDataset("mup_testing_vctk_data/", aug_prob=0,
                        test=False, segment_len=64000, dual_channel=False)
    dataloader = DataLoader(data, batch_size=16, shuffle=False, num_workers=2)
    # record data from the model activations over a few steps of training
    # this returns a pandas dataframe
    df = get_coord_data(models, dataloader, optimizer='adam', lr=3e-2, nsteps=7,
                        lossfn=F.mse_loss, nseeds=5, one_hot_target=False,
                        # filter_module_by_name=lambda name: 'out_conv' in name
                        )
    # This saves the coord check plots to filename.
    fig_mup = plot_coord_data(df, save_to="plot_mup_shallow_outlinear_v2.png",
                              subplot_width=10, subplot_height=15)


    def lazy_model_nomup(width):
        # `set_base_shapes` returns the model
        def get_model_nomup(width):
            model = VeryShallowMupAudioUNet(width, ismup=False)
            return model

        return lambda: get_model_nomup(width)
        # Note: any custom initialization with `mup.init` would need to
        # be done inside the lambda as well

    nomup_models = {2: lazy_model_nomup(2), 4: lazy_model_nomup(4), 8: lazy_model_nomup(8), 16: lazy_model_nomup(16), 32: lazy_model_nomup(32), 64: lazy_model_nomup(64), 128: lazy_model_nomup(128)}

    df = get_coord_data(nomup_models, dataloader, optimizer='adam', lr=3e-2, nsteps=7,
                        lossfn=F.mse_loss, nseeds=5, one_hot_target=False, mup=False
                        # filter_module_by_name=lambda name: 'out_conv' in name
                        )
    # This saves the coord check plots to filename.
    fig_mup = plot_coord_data(df, save_to="plot_nomup_shallow_outlinear_v2.png",
                              subplot_width=10, subplot_height=15)
