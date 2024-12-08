from attention_unet import AttentionUNet
from dense_unet import DenseUNet
from unet import Unet
from residual_unet import ResidualUNet
from unet_plusplus import UNetPlusPlus
from monai.networks.nets import UNETR

#model_1_name = 'unet'
#model_2_name = 'residualUnet'
#model_3_name = 'unetPlusplus'
#model_4_name = 'denseUnet'
#model_5_name = 'attentionUnet'
#all_model_name = [model_1_name, model_2_name, model_3_name, model_4_name, model_5_name]

model_6_name = 'monai'
model_7_name = 'unet'
all_model_name = [model_6_name, model_7_name]

#TODO: Add models
models = {}
#models[model_1_name] = Unet().to(device)
#models[model_2_name] = ResidualUNet().to(device)
#models[model_3_name] = UNetPlusPlus().to(device)
#models[model_4_name] = DenseUNet().to(device)
#models[model_5_name] = AttentionUNet().to(device)

models[model_6_name] = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(112, 112),  # 2D image size
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        spatial_dims=2,  # Using 2D dimensions
    )

models[model_7_name] = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(112, 112),  # 2D image size
        feature_size=32,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        spatial_dims=2,  # Using 2D dimensions
    )

default_model_paths = {}
default_model_paths[model_6_name] = "best_model.pth"
default_model_paths[model_7_name] = "unetR-dice-new.pth"