_base_ = './rtmdet-r_l_syncbn_fast_2xb4-36e_dota-ms.py'

checkpoint = 'Link Removeds://REMOVE_CVPR_POLICY

# ========================modified parameters======================
deepen_factor = 0.67
widen_factor = 0.75

# Submission dir for result submit
submission_dir = './work_dirs/{{fileBasenameNoExtension}}/submission'

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint)),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

# Inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     dataset=dict(
#         data_root=_base_.data_root,
#         ann_file='', # test set has no annotation
#         data_prefix=dict(img_path=_base_.test_data_prefix),
#         pipeline=_base_.test_pipeline))
# test_evaluator = dict(
#     type='mmrotate.DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix=submission_dir)
