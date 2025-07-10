# RemovePoliceCVPR
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from .img_prompt import *

@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.get_prompt = None
        self.get_prompt_flag = False
        self.img_prompt = None
        if 'img_prompt' in kwargs.keys():
            self.img_prompt = kwargs['img_prompt'] if kwargs['img_prompt'] else None

            width = kwargs['image_size_width']
            height = kwargs['image_size_height']
            psize = kwargs['prompt_size']
            kwargs.pop('image_size_width'), kwargs.pop('image_size_height'), kwargs.pop('prompt_size')
            
            if self.img_prompt == 'modprompt':
                modprompt_backbone = kwargs['modprompt_backbone']
                modprompt_encoder_depth = kwargs['modprompt_encoder_depth']
                modprompt_in_channels = kwargs['modprompt_in_channels']
                modprompt_out_channels = kwargs['modprompt_out_channels']
                modprompt_encoder_weights = kwargs['modprompt_encoder_weights']
                modprompt_alpha = kwargs['modprompt_alpha']
                
                kwargs.pop('modprompt_backbone'), kwargs.pop('modprompt_encoder_depth')
                kwargs.pop('modprompt_in_channels'), kwargs.pop('modprompt_out_channels')
                kwargs.pop('modprompt_encoder_weights'), kwargs.pop('modprompt_alpha')
                
            kwargs.pop('img_prompt')

        if 'image_size_width' in kwargs.keys():
            kwargs.pop('image_size_width')

        if 'image_size_height' in kwargs.keys():
            kwargs.pop('image_size_height')
            
        if 'prompt_size' in kwargs.keys():
            kwargs.pop('prompt_size')
        
        super().__init__(*args, **kwargs)
        
        if self.img_prompt == 'all':
            self.img_prompt = all_prompter(width=width, height=height, psize=psize)
        elif self.img_prompt == 'allscale':
            self.img_prompt = all_prompter_scale(width=width, height=height, psize=psize)
        elif self.img_prompt == 'conv':
            self.img_prompt = conv_prompter(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'fixed':
            self.img_prompt= fixed_patch(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'random':
            self.img_prompt = random_patch(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'padding':
            self.img_prompt = padding(width=width, height=height, psize=psize)
        elif self.img_prompt == 'modprompt':
            self.img_prompt = modprompt_prompter(backbone=modprompt_backbone,
                                            encoder_depth=modprompt_encoder_depth, 
                                            in_channels=modprompt_in_channels,
                                            out_channels=modprompt_out_channels,
                                            encoder_weights=modprompt_encoder_weights,
                                            alpha=modprompt_alpha)


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        
        
        if self.img_prompt is not None:
            batch_inputs = self.img_prompt(batch_inputs)
            if self.get_prompt_flag == True:
                self.get_prompt = batch_inputs
        
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        
        self.get_prompt = None
        self.get_prompt_flag = False
        self.img_prompt = None
        if 'img_prompt' in kwargs.keys():
            self.img_prompt = kwargs['img_prompt'] if kwargs['img_prompt'] else None

            width = kwargs['image_size_width']
            height = kwargs['image_size_height']
            psize = kwargs['prompt_size']
            kwargs.pop('image_size_width'), kwargs.pop('image_size_height'), kwargs.pop('prompt_size')
            
            if self.img_prompt == 'modprompt':
                modprompt_backbone = kwargs['modprompt_backbone']
                modprompt_encoder_depth = kwargs['modprompt_encoder_depth']
                modprompt_in_channels = kwargs['modprompt_in_channels']
                modprompt_out_channels = kwargs['modprompt_out_channels']
                modprompt_encoder_weights = kwargs['modprompt_encoder_weights']
                modprompt_alpha = kwargs['modprompt_alpha']
                
                kwargs.pop('modprompt_backbone'), kwargs.pop('modprompt_encoder_depth')
                kwargs.pop('modprompt_in_channels'), kwargs.pop('modprompt_out_channels')
                kwargs.pop('modprompt_encoder_weights'), kwargs.pop('modprompt_alpha')
                
            kwargs.pop('img_prompt')

        if 'image_size_width' in kwargs.keys():
            kwargs.pop('image_size_width')

        if 'image_size_height' in kwargs.keys():
            kwargs.pop('image_size_height')
            
        if 'prompt_size' in kwargs.keys():
            kwargs.pop('prompt_size')
        
        super().__init__(*args, **kwargs)

        if self.img_prompt == 'all':
            self.img_prompt = all_prompter(width=width, height=height, psize=psize)
        elif self.img_prompt == 'allscale':
            self.img_prompt = all_prompter_scale(width=width, height=height, psize=psize)
        elif self.img_prompt == 'conv':
            self.img_prompt = conv_prompter(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'fixed':
            self.img_prompt= fixed_patch(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'random':
            self.img_prompt = random_patch(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'padding':
            self.img_prompt = padding(width=width, height=height, psize=psize)
        elif self.img_prompt == 'modprompt':
            self.img_prompt = modprompt_prompter(backbone=modprompt_backbone,
                                            encoder_depth=modprompt_encoder_depth, 
                                            in_channels=modprompt_in_channels,
                                            out_channels=modprompt_out_channels,
                                            encoder_weights=modprompt_encoder_weights,
                                            alpha=modprompt_alpha)
  

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                
                if len(embedding_path[0]) > 1: # Problem with v1? 
                    embedding_path = embedding_path[0]
                
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features

        if self.img_prompt is not None:
            batch_inputs = self.img_prompt(batch_inputs)
            if self.get_prompt_flag == True:
                self.get_prompt = batch_inputs
        
        
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
    
    
    

@MODELS.register_module()
class TRESYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter        
        self.get_prompt = None
        self.get_prompt_flag = False
        self.img_prompt = None
        if 'img_prompt' in kwargs.keys():
            self.img_prompt = kwargs['img_prompt'] if kwargs['img_prompt'] else None

            width = kwargs['image_size_width']
            height = kwargs['image_size_height']
            psize = kwargs['prompt_size']
            kwargs.pop('image_size_width'), kwargs.pop('image_size_height'), kwargs.pop('prompt_size')
            
            if self.img_prompt == 'modprompt':
                modprompt_backbone = kwargs['modprompt_backbone']
                modprompt_encoder_depth = kwargs['modprompt_encoder_depth']
                modprompt_in_channels = kwargs['modprompt_in_channels']
                modprompt_out_channels = kwargs['modprompt_out_channels']
                modprompt_encoder_weights = kwargs['modprompt_encoder_weights']
                modprompt_alpha = kwargs['modprompt_alpha']
                
                kwargs.pop('modprompt_backbone'), kwargs.pop('modprompt_encoder_depth')
                kwargs.pop('modprompt_in_channels'), kwargs.pop('modprompt_out_channels')
                kwargs.pop('modprompt_encoder_weights'), kwargs.pop('modprompt_alpha')
                
            kwargs.pop('img_prompt')

        if 'image_size_width' in kwargs.keys():
            kwargs.pop('image_size_width')

        if 'image_size_height' in kwargs.keys():
            kwargs.pop('image_size_height')
            
        if 'prompt_size' in kwargs.keys():
            kwargs.pop('prompt_size')        
        
        super().__init__(*args, **kwargs)
        
        if self.img_prompt == 'all':
            self.img_prompt = all_prompter(width=width, height=height, psize=psize)
        elif self.img_prompt == 'allscale':
            self.img_prompt = all_prompter_scale(width=width, height=height, psize=psize)
        elif self.img_prompt == 'conv':
            self.img_prompt = conv_prompter(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'fixed':
            self.img_prompt= fixed_patch(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'random':
            self.img_prompt = random_patch(width=width, height=height, psize=psize)
        elif self.img_prompt  == 'padding':
            self.img_prompt = padding(width=width, height=height, psize=psize)
        elif self.img_prompt == 'modprompt':
            self.img_prompt = modprompt_prompter(backbone=modprompt_backbone,
                                            encoder_depth=modprompt_encoder_depth, 
                                            in_channels=modprompt_in_channels,
                                            out_channels=modprompt_out_channels,
                                            encoder_weights=modprompt_encoder_weights,
                                            alpha=modprompt_alpha)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                
                if len(embedding_path[0]) > 1: # Problem with v1? 
                    embedding_path = embedding_path[0]
                
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
                
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            self.embeddings.requires_grad = False

            self.text_feature_residuals = nn.Parameter(torch.zeros_like(self.embeddings))
            self.alpha = 1.
            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        
        if self.img_prompt is not None:
            batch_inputs = self.img_prompt(batch_inputs)
            if self.get_prompt_flag == True:
                self.get_prompt = batch_inputs

        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None] + self.alpha * self.text_feature_residuals
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
