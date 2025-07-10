_base_ = './ppyoloe_s_fast_8xb32-300e_coco.py'

# The pretrained model is geted and converted from official PPYOLOE.
# Link Removeds://RemoveLinkCVPR
checkpoint = 'Link Removeds://REMOVE_CVPR_POLICY

deepen_factor = 1.33
widen_factor = 1.25

train_batch_size_per_gpu = 16

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
