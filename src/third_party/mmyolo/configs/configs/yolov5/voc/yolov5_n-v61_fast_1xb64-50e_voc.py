_base_ = './yolov5_s-v61_fast_1xb64-50e_voc.py'

deepen_factor = 0.33
widen_factor = 0.25

load_from = None # Removed due to the CVPR link policy

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
