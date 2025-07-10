import torch
import torch.nn as nn
import numpy as np
from .translator import *


class ModPromptPrompt(nn.Module):
    def __init__(self, backbone='resnet34', encoder_depth=5, in_channels=3,out_channels=3,encoder_weights='imagenet',alpha=1.0):
        super(ModPromptPrompt, self).__init__()
        
        self.translator = Translator(
            backbone=backbone, 
            encoder_depth=encoder_depth, 
            in_channels=in_channels,
            out_channels=out_channels, 
            encoder_weights=encoder_weights,
        )
        
        self.alpha = alpha

    def forward(self, x):
        prompt = self.translator(x)
        return torch.clamp(x + self.alpha*prompt, min=0.0, max=1.0)



class ConvPrompt(nn.Module):
    def __init__(self, width, height, psize, scale_values=0.001):
        super(ConvPrompt, self).__init__()
        
        self.isize_width = width
        self.isize_height = height
        self.conv = nn.Conv2d(3, 3, (3, 3), stride=(1, 1), padding=(1, 1))
        self.alpha = 1.0

    def forward(self, x):
        prompt = self.conv(x)
        return torch.clamp(x + self.alpha*prompt, min=0.0, max=1.0)



class AllPrompterScale(nn.Module):
    def __init__(self, width, height, psize, scale_values=0.001):
        super(AllPrompterScale, self).__init__()
        
        self.isize_width = height
        self.isize_height = width
        self.all_prompt_patch = nn.Parameter(torch.randn([1, 3,  self.isize_width, self.isize_height]) * scale_values)
        self.all_scale_patch = nn.Parameter(torch.ones([1, 3,  self.isize_width, self.isize_height]))
        self.eps = 1e-6

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize_width, self.isize_height]).cuda()
        prompt[:, :, :self.isize_width, :self.isize_height] = self.all_prompt_patch
        return torch.clamp((x/(self.all_scale_patch + self.eps)) + prompt, min=0.0, max=1.0)



class AllPrompter(nn.Module):
    def __init__(self, width, height, psize, scale_values=0.001):
        super(AllPrompter, self).__init__()
        
        self.isize_width = height
        self.isize_height = width
        self.all_prompt_patch = nn.Parameter(torch.randn([1, 3,  self.isize_width, self.isize_height]) * scale_values)

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize_width, self.isize_height]).cuda()
        prompt[:, :, :self.isize_width, :self.isize_height] = self.all_prompt_patch
        return torch.clamp(x + prompt, min=0.0, max=1.0)


class FixedPatchPrompter(nn.Module):
    def __init__(self, width, height, psize):
        super(FixedPatchPrompter, self).__init__()
        
        self.isize_width = width
        self.isize_height = height
        self.psize = psize
        
        self.fixed_patch = nn.Parameter(torch.randn([1, 3,  self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize_height, self.isize_width]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.fixed_patch
        return x + prompt
    
    

class RandomPatchPrompter(nn.Module):
    def __init__(self, width, height, psize):
        super(RandomPatchPrompter, self).__init__()
        
        self.isize_width = height
        self.isize_height = width
        self.psize = psize
        
        self.random_patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize_width - self.psize)
        y_ = np.random.choice(self.isize_height - self.psize)

        prompt = torch.zeros([1, 3, self.isize_width, self.isize_height]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.random_patch

        return x + prompt



class PadPrompter(nn.Module):
    def __init__(self, width, height, psize):
        super(PadPrompter, self).__init__()

        image_size_width = height
        image_size_height = width
        self.psize  = psize
    
        self.base_size_width = image_size_width - self.psize * 2
        self.base_size_height = image_size_height - self.psize * 2
        
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.psize, image_size_height]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.psize, image_size_height]))
        
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size_width - self.psize*2, self.psize]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size_width - self.psize*2, self.psize]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size_width, self.base_size_height).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        return x + prompt


def modprompt_prompter(backbone='resnet34', encoder_depth=5, in_channels=3,out_channels=3,encoder_weights='imagenet',alpha=1.0):
    return ModPromptPrompt(backbone=backbone, 
                       encoder_depth=encoder_depth, 
                       in_channels=in_channels,
                       out_channels=out_channels,
                       encoder_weights=encoder_weights,
                       alpha=alpha)


def conv_prompter(width, height, psize):
    return ConvPrompt(width=width, height=height, psize=psize)


def all_prompter(width, height, psize):
    return AllPrompter(width=width, height=height, psize=psize)


def all_prompter_scale(width, height, psize):
    return AllPrompterScale(width=width, height=height, psize=psize)


def padding(width, height, psize):
    return PadPrompter(width=width, height=height, psize=psize)


def fixed_patch(width, height, psize):
    return FixedPatchPrompter(width=width, height=height, psize=psize)


def random_patch(width, height, psize):
    return RandomPatchPrompter(width=width, height=height, psize=psize)