import os
import cv2
import argparse
import os.path as osp
import re
import copy
import numpy as np

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import supervision as sv

COLOR=sv.Color(255, 255, 0) # yellow
#COLOR=sv.Color(255, 0, 0) # red
thickness=7 # LLVIP or thickness=3 flir
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=7, color=COLOR)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1, 
                                 color=COLOR
                                 )


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--image', help='image path, include image file or dir.')
    parser.add_argument(
        '--text',
        help=
        'text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    
    parser.add_argument('--exp-name', help='Additional Name for the experiment')
    parser.add_argument('--seed', type=int, default=123, help='Seed value')    
    
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--threshold',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--show',
                        action='store_true',
                        help='show the detection results.')
    parser.add_argument('--scores',
                    action='store_true',
                    help='show the scores.')
    
    parser.add_argument(
        '--annotation',
        action='store_true',
        help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference_detector(model,
                       image,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       output_dir='./work_dir',
                       use_amp=False,
                       show=False,
                       annotation=False,
                       dataset='llvip'):
    
    data_info = dict(img_id=0, img_path=image, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    model.get_prompt_flag = False if 'zeroshot' in output_dir else True
    model.get_prompt_flag = False
    prompted_img = None
    
    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
                
        if model.get_prompt_flag:
            prompted_img = copy.deepcopy(model.get_prompt.squeeze().cpu().numpy())
            prompted_img = np.transpose(prompted_img * 255, (1, 2, 0))
            prompted_img = cv2.cvtColor(prompted_img, cv2.COLOR_BGR2RGB)
        
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None

    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)

    print("-------------------------------")
    print("-------------------------------")
    print("-------------------------------")
    print(image_path)
    print(detections)
    print("-------------------------------")
    print("-------------------------------")
    print("-------------------------------")
    # # ds = sv.DetectionDataset.from_coco("./data/FLIR_aligned/test", annotations_path="./data/FLIR_aligned/flir_ir_test_full.json", force_masks=False)
    # ds = sv.DetectionDataset.from_coco("./data/LLVIP/infrared/test", annotations_path="./data/LLVIP/infrared/llvip_ir_test.json", force_masks=False)
    # # box_annotator = sv.BoxAnnotator()
    # box_annotator = sv.BoundingBoxAnnotator(thickness=7, color=COLOR)
    # label_annotator = sv.LabelAnnotator()
    # annotated_images = []
    
    # LLVIP_SIZE=3463 # '230120.jpg'
    # # FLIR_SIZE       # 09479.jpg   
    # for i in range(LLVIP_SIZE):
    #     path, image, annotations = ds[i]
    #     print(path.split('/')[-1])
    #     if path.split('/')[-1] == '230120.jpg':
    #         print(path)
    #         labels = [ds.classes[class_id] for class_id in annotations.class_id]
            
    #         annotated_image = image.copy()
    #         annotated_image = box_annotator.annotate(annotated_image, annotations)
    #         # annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
    #         cv2.imwrite(osp.join(output_dir, osp.basename(path.split('/')[-1]).replace('.jpg', '.png')), annotated_image)
            
    #         break
    #     # annotated_images.append(annotated_image)




    #     # cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), grid)
    
    # exit()




    # labels = []
    # if args.scores:
    #     labels = [
    #         f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
    #         zip(detections.class_id, detections.confidence) 
    #     ]
    # else:
    #     labels = [f"{texts[class_id][0]}" for class_id in detections.class_id]
    

    # print(pred_instances['labels'])
    # print(detections.class_id)
    
    
    # label images
    if prompted_img is not None:
        image = prompted_img
    else:
        image = cv2.imread(image_path)
        anno_image = image.copy()
    
    
    if dataset == "llvip":
        up_width = 1280
        up_height = 1024
    elif dataset == "flir":
        up_width = 640
        up_height = 512
    
    
    # up_points = (up_width, up_height)

    # image = cv2.resize((image * 255).astype(np.uint8), up_points, interpolation= cv2.INTER_LINEAR)

    
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    # image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
        
    # print(osp.join(output_dir, osp.basename(image_path)))
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path.replace('.jpg', '.png'))), image)
    # exit()


if __name__ == '__main__':
    args = parse_args()

    seed_everything(args.seed)

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    
    if args.exp_name is not None:
        cfg.work_dir = cfg.work_dir + '_' +  str(args.exp_name)
    

    if not(args.checkpoint.endswith(".pth")):
        # get best checkpoint
        res = [x for x in os.listdir(args.checkpoint) if re.search("best", x)]
        args.checkpoint = osp.join(args.checkpoint, res[0])

    cfg.load_from = None # Removed due to the CVPR link policy
    
    # init model
    cfg.load_from = None # Removed due to the CVPR link policy
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg')
        ]
    else:
        images = [args.image]

    dataset = args.config.split('/')[-2]

    # reparameterize texts
    model.reparameterize(texts)
    progress_bar = ProgressBar(len(images))
    for idx, image_path in enumerate(images):
        inference_detector(model=model,
                           image=image_path,
                           texts=texts,
                           test_pipeline=test_pipeline,
                           max_dets=args.topk,
                           score_thr=args.threshold,
                           output_dir=output_dir,
                           use_amp=args.amp,
                           show=args.show,
                           annotation=args.annotation,
                           dataset=dataset)
        
        progress_bar.update()
        if(idx >= 50 and 'flir' in dataset):
            break
        if(idx >= 20 and 'llvip' in dataset):
            break
