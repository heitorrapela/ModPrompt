_base_ = ('./nyu_base/nyu_base_s_prompt_base.py')



custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 19
num_training_classes = 19
metainfo = dict(classes=["bathtub", "bed", "bookshelf", "box", "chair", "counter", "desk", "door", "dresser", "garbage bin", "lamp", "monitor", "night stand", "pillow", "sink", "sofa", "table", "television", "toilet"])

max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 8
load_from = None # Removed due to the CVPR link policy
persistent_workers = False

# model settings
model = dict(type='TRESYOLOWorldDetector',
        mm_neck=True,
        img_prompt='random',
        prompt_size=30,
        num_train_classes=num_training_classes,
        num_test_classes=num_classes,
        image_size_width=_base_.img_scale[0],
        image_size_height=_base_.img_scale[1],
        embedding_path='embeddings/nyu_v2_clip_projections.npy',
        prompt_dim=text_channels,
        num_prompts=19,
        freeze_prompt=False,
        data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
        backbone=dict(_delete_=True,
                    type='MultiModalYOLOBackbone',
                    text_model=None,
                    image_model={{_base_.model.backbone}},
                    frozen_stages=4,
                    with_text_model=False),
        neck=dict(type='YOLOWorldPAFPN',
                freeze_all=True,
                guide_channels=text_channels,
                embed_channels=neck_embed_channels,
                num_heads=neck_num_heads,
                block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
        bbox_head=dict(type='YOLOWorldHead',
                head_module=dict(type='YOLOWorldHeadModule',
                freeze_all=True,
                use_bn_head=True,
                embed_dims=text_channels,
                num_classes=num_training_classes)),
        train_cfg=dict(assigner=dict(num_classes=num_training_classes)))


# dataset settings
nyu_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        metainfo=metainfo,
        type='YOLOv5CocoDataset',
        data_root='data/nyu_v2/',
        ann_file='train_gt_box.json',
        data_prefix=dict(img='nyuv2_train_color/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/nyu_v2_class_text.json',
    pipeline=_base_.train_pipeline)


train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=nyu_train_dataset)


nyu_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5CocoDataset',
                 metainfo=metainfo,
                 data_root='data/nyu_v2/',
                 test_mode=True,
                 ann_file='test_gt_box.json',
                 data_prefix=dict(img='nyuv2_test_color/'),
                 batch_shapes_cfg=None),
    class_text_path='data/texts/nyu_v2_class_text.json',
    pipeline=_base_.test_pipeline)

val_dataloader = dict(dataset=nyu_val_dataset)
test_dataloader = val_dataloader
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/nyu_v2/test_gt_box.json',
    metric='bbox')

test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2)
]

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                        'backbone.text_model':
                                        dict(lr_mult=0.01),
                                        'logit_scale':
                                        dict(weight_decay=0.0),
                                        'embeddings':
                                        dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')
