_base_ = './yolov5_s-v61_fast_1xb64-50e_voc.py'

deepen_factor = 1.0
widen_factor = 1.0
train_batch_size_per_gpu = 32
train_num_workers = 8

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

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu, num_workers=train_num_workers)

optim_wrapper = dict(
    optimizer=dict(batch_size_per_gpu=train_batch_size_per_gpu))
