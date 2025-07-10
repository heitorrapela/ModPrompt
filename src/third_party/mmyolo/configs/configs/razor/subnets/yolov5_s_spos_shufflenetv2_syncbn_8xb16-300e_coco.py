_base_ = [
    'mmrazor::_base_/nas_backbones/spos_shufflenet_supernet.py',
    '../../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
]

checkpoint_file = 'Link Removeds://REMOVE_CVPR_POLICY
fix_subnet = 'Link Removeds://REMOVE_CVPR_POLICY
widen_factor = 1.0
channels = [160, 320, 640]

_base_.nas_backbone.out_indices = (1, 2, 3)
_base_.nas_backbone.init_cfg = dict(
    type='Pretrained',
    checkpoint=checkpoint_file,
    prefix='architecture.backbone.')
nas_backbone = dict(
    type='mmrazor.sub_model',
    fix_subnet=fix_subnet,
    cfg=_base_.nas_backbone,
    extra_prefix='architecture.backbone.')

_base_.model.backbone = nas_backbone
_base_.model.neck.widen_factor = widen_factor
_base_.model.neck.in_channels = channels
_base_.model.neck.out_channels = channels
_base_.model.bbox_head.head_module.in_channels = channels
_base_.model.bbox_head.head_module.widen_factor = widen_factor

find_unused_parameters = True
