_base_ = './yolov6_m_fast_1xb32-100e_ionogram.py'

# ======================= Modified parameters =======================
# -----model related-----
deepen_factor = 1
widen_factor = 1

# -----train val related-----
load_from = None # Removed due to the CVPR link policy

# ====================== Unmodified in most cases ===================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_ratio=1. / 2,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_ratio=1. / 2,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        block_act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
