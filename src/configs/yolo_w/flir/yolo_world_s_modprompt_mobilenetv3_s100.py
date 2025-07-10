# yolo_world_s_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py
_base_ = ('flir_base.py')

custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 3 ## person : 0, bicycle: 1, car: 2
num_training_classes = 3
metainfo = dict(classes=["person","bicycle","car"])
max_epochs = 80  # Maximum training epochs # hparams from original repo 80 
close_mosaic_epochs=10
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 8
load_from='pretrained_models/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained.pth'


# model settings
model = dict(
    type='YOLOWorldDetector',
    img_prompt='modprompt',
    modprompt_backbone='timm-mobilenetv3_small_minimal_100',
    modprompt_encoder_depth=5,
    modprompt_in_channels=3,
    modprompt_out_channels=3,
    modprompt_encoder_weights='imagenet',
    modprompt_alpha=1.0,
    prompt_size=30,
    image_size_width=_base_.img_scale[0],
    image_size_height=_base_.img_scale[1],
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        # frozen_stages=4,
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldDualPAFPN',
              # freeze_all=True,
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              text_enhancder=dict(type='ImagePoolingAttentionModule',
                                  embed_channels=256,
                                  num_heads=8)),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    embed_dims=text_channels,
                                    #freeze_all=True,
                                    num_classes=num_training_classes)),
    
                                    

    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]


flir_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        metainfo=metainfo,
        type='YOLOv5CocoDataset',
        data_root='data/FLIR_aligned/',
        ann_file='flir_ir_train_full.json',
        data_prefix=dict(img='JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/flir_class_text.json',
    pipeline=train_pipeline)


train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=flir_train_dataset)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

flir_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5CocoDataset',
                 metainfo=metainfo,
                 data_root='data/FLIR_aligned/',
                 test_mode=True,
                 ann_file='flir_ir_test_full.json',
                 data_prefix=dict(img='JPEGImages/'),
                 batch_shapes_cfg=None),
    class_text_path='data/texts/flir_class_text.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=flir_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/FLIR_aligned/flir_ir_test_full.json',
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
         switch_pipeline=train_pipeline_stage2)
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
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')
