import torch
import torch.nn as nn
import shutil
from vit_pytorch import SimpleViT
import os
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import models
from torchsummary import summary

from path import *

class ViTWrapper():
    def __init__(self, name = 'vit', device = 'cpu', classes = 2, pretrain = False, pen_features : int = 0):
        self.model = SimpleViT(
            image_size = 1024,
            patch_size = 32,
            num_classes = classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            channels = 1
        )
        self.name = name
        self.cls = classes
        self.device = device
        self.model.linear_head = nn.Identity()
        self.pen_features = pen_features
        if not pretrain:
            self.load_state(s = 'vit_model_best.pth')
            self.model.linear_head = self.__set_head(classes)
        self.model = self.model.to(device)
        return
    
    def __set_head(self, cls):
        return nn.Sequential(
            nn.Linear(1024, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, cls)
        )

    def binary(self):
        self.cls = 1
        self.model.linear_head[4] = nn.Linear(100, 1)
        self.model.to(self.device)
    
    def set_csv_model(self, base):
        if self.pen_features != 0:
            self.model = ModelCSV(base, self.model, self.pen_features, self.cls, self.device)
            print("CSV Model!")
        else:
            print("Cannot use CSV Model! No Pen Features loaded.")
        return
    
    def get_model(self):
        return self.model
        
    def save_state(self, state, is_best):
        out = os.path.join(CHECKPOINTS, f'{self.name}_checkpoint.pth')
        torch.save(state, out)
        if is_best:
            shutil.copyfile(out, os.path.join(CHECKPOINTS,f'{self.name}_model_best.pth'))
    
    def load_state(self, s):
        s = torch.load(os.path.join(CHECKPOINTS, s))
        self.model.load_state_dict(s['state_dict'])
    
    def resume(self, r):
        print(f"=> loading checkpoint '{r}'")
        c = torch.load(os.path.join(CHECKPOINTS, r))
        self.load_state(c['state_dict'])
        return c['epoch'], c['best_val_loss'], c['exit_counter'], c['optimizer'], c['best_val_f1']

class ResnetWrapper():
    def __init__(self, name : str = 'resnet18', device : str = 'cpu', classes : int = 2, pretrain : bool = False, pen_features : int = 0):
        self.model = resnet18(ResNet18_Weights.DEFAULT)
        self.name = name
        self.device = device
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Identity()
        self.pen_features = pen_features
        self.cls = classes
        if not pretrain:
            self.load_state(s = 'resnet18_model_best.pth')
            self.model.fc = self.__set_head(classes)
        self.model = self.model.to(device)
        return
    
    def __set_head(self, cls):
        return nn.Sequential(
            nn.Linear(512, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, cls)
        )

    def binary(self):
        self.cls = 1
        self.model.fc[4] = nn.Linear(100, 1)
        self.model.to(self.device)

    def freeze(self):
        print("BRRRRRRRRRRRRRRRRR!")
        for name, param in self.model.named_parameters():
            if 'model.fc' in name: continue
            if 'classify' in name: continue
            param.requires_grad = False
    
    def get_model(self):
        return self.model
    
    def set_csv_model(self, base):
        if self.pen_features != 0:
            self.model = ModelCSV(base, self.model, self.pen_features, self.cls, self.device)
            print("CSV Model!")
        else:
            print("Cannot use CSV Model! No Pen Features loaded.")
        return
        
    def save_state(self, state, is_best):
        out = os.path.join(CHECKPOINTS, f'{self.name}_checkpoint.pth')
        torch.save(state, out)
        if is_best:
            shutil.copyfile(out, os.path.join(CHECKPOINTS,f'{self.name}_model_best.pth'))
    
    def load_state(self, s):
        s = torch.load(os.path.join(CHECKPOINTS, s))
        self.model.load_state_dict(s['state_dict'])
    
    def resume(self, r):
        print(f"=> loading checkpoint '{r}'")
        c = torch.load(os.path.join(CHECKPOINTS, r))
        self.load_state(c['state_dict'])
        return c['epoch'], c['best_val_loss'], c['exit_counter'], c['optimizer']

class ModelCSV(nn.Module):
    def __init__(self, name, model, pen_features, cls, device):
        super(ModelCSV, self).__init__()
        self.model = model
        self.device = device
        if name == 'resnet': 
            self.model.fc[4] = nn.Identity()
        elif name == 'vit': 
            self.model.linear_head[4] = nn.Identity()
        self.classify = nn.Linear(100 + pen_features, cls)
        self.model.to(device), self.classify.to(device)
    
    def forward(self, img, pfeat):
        x = self.model(img)
        x = torch.cat((x, pfeat), dim=1).to(self.device)
        return self.classify(x)

class EncoderFactory(nn.Module):
    def __init__(self, backbone_name='resnet18', device='cuda:0'):
        super().__init__()
        self.device = device
        self.backbone_name = backbone_name.lower()
        self.name = backbone_name 
        
        # 1. Initialize the backbone
        if self.backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights)
            self._fix_first_layer_resnet(base_model) # <--- Fix 1-channel input
            # Remove fc layer
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.embed_dim = 512
            
        elif self.backbone_name == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT
            base_model = models.resnet34(weights=weights)
            self._fix_first_layer_resnet(base_model) # <--- Fix 1-channel input
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.embed_dim = 512

        elif self.backbone_name == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            base_model = models.mobilenet_v3_small(weights=weights)
            self._fix_first_layer_generic(base_model.features) # <--- Fix 1-channel input
            self.feature_extractor = nn.Sequential(base_model.features, base_model.avgpool)
            self.embed_dim = 576

        elif self.backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT
            base_model = models.efficientnet_b0(weights=weights)
            self._fix_first_layer_generic(base_model.features) # <--- Fix 1-channel input
            self.feature_extractor = nn.Sequential(base_model.features, base_model.avgpool)
            self.embed_dim = 1280

        else:
            raise ValueError(f"Backbone {self.backbone_name} not supported.")
        
        self.to(self.device)

    def _fix_first_layer_resnet(self, model):
        # ResNet's first layer is named 'conv1'
        old_layer = model.conv1
        new_layer = nn.Conv2d(1, old_layer.out_channels, 
                              kernel_size=old_layer.kernel_size, 
                              stride=old_layer.stride, 
                              padding=old_layer.padding, 
                              bias=old_layer.bias is not None)
        # Sum the weights of the RGB channels to preserve learned patterns
        with torch.no_grad():
            new_layer.weight[:] = torch.sum(old_layer.weight, dim=1, keepdim=True)
        model.conv1 = new_layer

    def _fix_first_layer_generic(self, features):
        # MobileNet/EfficientNet store the first layer in features[0][0]
        old_layer = features[0][0]
        new_layer = nn.Conv2d(1, old_layer.out_channels, 
                              kernel_size=old_layer.kernel_size, 
                              stride=old_layer.stride, 
                              padding=old_layer.padding, 
                              bias=old_layer.bias is not None)
        with torch.no_grad():
            new_layer.weight[:] = torch.sum(old_layer.weight, dim=1, keepdim=True)
        features[0][0] = new_layer

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.flatten(start_dim=1)

    def save_state(self, state, is_best):
        out = os.path.join(CHECKPOINTS, f'{self.name}_checkpoint.pth')
        torch.save(state, out)
        if is_best:
            shutil.copyfile(out, os.path.join(CHECKPOINTS,f'{self.name}_model_best.pth'))
        
    def load_state(self, filename):
        path = os.path.join(CHECKPOINTS, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                self.load_state_dict(checkpoint)
            print(f"Loaded weights from {filename}")
        else:
            print(f"No checkpoint found at {filename}, starting fresh.")

    def get_model(self):
        return self