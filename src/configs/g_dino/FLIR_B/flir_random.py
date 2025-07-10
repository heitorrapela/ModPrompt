_base_ = [
    '../FLIR/flir_random.py',
]

load_from = None # Removed due to the CVPR link policy
model = dict(
    type='GroundinDINO',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(in_channels=[256, 512, 1024]),
)
