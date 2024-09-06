from llava.model.multimodal_encoder.sam2_hiera import Hiera
import torch

hiera = Hiera(
    embed_dim=144,
    num_heads=2,
    stages=[2, 6, 36, 4],
    global_att_blocks=[23, 33, 43],
    window_pos_embed_bkg_spatial_size=[7, 7],
    window_spec=[8, 4, 16, 8],
)

# a = torch.randn(1, 3, 224, 224)
# a = torch.randn(1, 3, 576, 576)
a = torch.randn(1, 3, 1024, 1024)
with torch.inference_mode():
    o = hiera(a)

for bb in o:
    print(bb.shape)
