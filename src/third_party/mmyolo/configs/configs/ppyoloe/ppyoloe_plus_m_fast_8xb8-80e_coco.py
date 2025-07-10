_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco.py'

# The pretrained model is geted and converted from official PPYOLOE.
# Link Removeds://RemoveLinkCVPR
load_from = None # Removed due to the CVPR link policy

deepen_factor = 0.67
widen_factor = 0.75

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
