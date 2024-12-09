from unet import Unet
from monai.networks.nets import UNETR
from monai.networks.nets import BasicUNetPlusPlus
from monai.networks.nets import DynUNet
from monai.networks.nets import BasicUNet

model_1_name = 'unetr'
model_2_name = 'basicunet'
model_3_name = 'unetplusplus'
model_4_name = 'dynunet'
all_model_name = [model_1_name, model_2_name, model_3_name, model_4_name]

#TODO: Add models
models = {}
models[model_1_name] = UNETR(
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
models[model_2_name] = BasicUNet(
        spatial_dims=2,  # 2D images
        in_channels=1,   # Single-channel grayscale input
        out_channels=2,  # Binary segmentation (two classes)
        features=(32, 64, 128, 256, 512, 64),
    )
models[model_3_name] = BasicUNetPlusPlus(
        spatial_dims=2,  # 2D images
        in_channels=1,   # Grayscale images
        out_channels=2,  # Binary segmentation
        features=(32, 64, 128, 256, 512, 64),  # Customize based on dataset complexity
    )
models[model_4_name] = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        kernel_size=[(3, 3), (3, 3), (3, 3)],
        strides=[(1, 1), (2, 2), (2, 2)],
        upsample_kernel_size=[(2, 2), (2, 2)],
        filters=[32, 64, 128] ,
        dropout=None,
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=False,  # Use True if deep supervision is required
        res_block=True          # Enable residual connections
    )

default_model_paths = {}
default_model_paths[model_1_name] = "unetr.pth"
default_model_paths[model_2_name] = "basicunet.pth"
default_model_paths[model_3_name] = "unetplusplus.pth"
default_model_paths[model_4_name] = "dynunet.pth"