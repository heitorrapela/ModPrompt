# To run the code

## You need to install the requirements (Or go to the original installation of yolo-world)

> cd requirements
> pip install -r basic_requirements.txt

## Create the pretrained_models folder with the pretrained weights
## (Download the pretrain weights from the original repo) and add inside pretrained_models

mkdir pretrained_models

## Add your dataset inside ./data folder (FLIR_aligned, LLVIP, nyu_v2)

## For YOLO World

> python tools/train_yolo.py configs/yolo_w/flir/yolo_world_s_modprompt_resnet34.py --work-dir output/

## For Grounding DINO

> python -m tools.train_gd configs/g_dino/FLIR/flir_modprompt_resnet.py --work-dir output/


# Useful pretrained weights for YOLO-World

yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained
yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea
yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained

# ADD inside the pretrained_models folder

> pretrained_models/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained.pth