import torch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print('Error importing segmentation_models_pytorch')
    print('Please install the package using pip install segmentation_models_pytorch==0.3.3')
    raise


class Translator(torch.nn.Module):
    
    def __init__(self, backbone='resnet34', encoder_depth=5, in_channels=3, out_channels=3, 
                 encoder_weights='imagenet', frozen=False, pretrained=True):
        
        super().__init__()
        
        self.translator = smp.Unet(encoder_name=backbone,
                                encoder_depth=encoder_depth,
                                encoder_weights=encoder_weights if pretrained else None,
                                in_channels=in_channels,
                                classes=out_channels)
        self.translator.segmentation_head[-1] = torch.nn.Sigmoid()
        
        self.frozen = frozen
        
        if self.frozen:
            self.freeze()
        else:
            self.unfreeze()
            
    
    def freeze(self):
        for param in self.translator.parameters():
            param.requires_grad = False
            
            
    def unfreeze(self):
        for param in self.translator.parameters():
            param.requires_grad = True
            
            
    def get_parameters(self):
        return self.translator.parameters()
    
    
    def forward(self, x):
        return self.translator(x)
