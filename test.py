from llava.model.multimodal_encoder.sam2_hiera import Hiera
import torch
from loguru import logger


def load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("image_encoder.trunk."):
                new_sd[k.replace("image_encoder.trunk.", "")] = v
        # print(new_sd.keys())
        missing_keys, unexpected_keys = model.load_state_dict(new_sd)
        if missing_keys:
            logger.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logger.error(unexpected_keys)
            raise RuntimeError()
        logger.info("Loaded checkpoint sucessfully")


hiera = Hiera(
    embed_dim=144,
    num_heads=2,
    stages=[2, 6, 36, 4],
    global_att_blocks=[23, 33, 43],
    window_pos_embed_bkg_spatial_size=[7, 7],
    window_spec=[8, 4, 16, 8],
)

load_checkpoint(hiera, "vendor/segment-anything-2/checkpoints/sam2_hiera_large.pt")

# a = torch.randn(1, 3, 224, 224)
# a = torch.randn(1, 3, 576, 576)
# a = torch.randn(1, 3, 2048, 2048)
# a = torch.randn(1, 3, 1024, 1280)
# a = torch.randn(1, 3, 800, 800)
a = torch.randn(1, 3, 896, 896)
with torch.inference_mode():
    o = hiera(a)

for bb in o:
    print(bb.shape)

# 1024x1024 outputs 1024 tokens
# 2048x2048 outputs 4096 tokens
# 1024x1280 outputs 1280 tokens
# 800x800 outputs 625 tokens
# 896x896 outputs 784 tokens
