_base_ = './rtmdet-r_l_syncbn_fast_2xb4-36e_dota-ms.py'

load_from = None # Removed due to the CVPR link policy

# Submission dir for result submit
submission_dir = './work_dirs/{{fileBasenameNoExtension}}/submission'

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
