import tensorflow as tf
import torch.nn as nn
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
import random
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import torchvision
import matplotlib.patches as patches
from itertools import cycle

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)
            
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats= 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential
                (CNNBlock(channels, channels//2, kernel_size =1),
                CNNBlock(channels//2, channels, kernel_size =3, padding =1)),
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (num_classes + 5)*3, bn_act=False, kernel_size=1)
        ) 
        self.num_classes = num_classes
        
    def forward(self, x):
        return (self.pred(x)
                .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2))
        
class Yolov3LoadPretrain(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _create_conv_layers(self):
        config = [
            (32, 3, 1),
            (64, 3, 2),
            ["B", 1],
            (128, 3, 2),
            ["B", 2],
            (256, 3, 2),
            ["B", 8],
            (512, 3, 2),
            ["B", 8],
            (1024, 3, 2),
            ["B", 4],  # Đến đây là Darknet-53
        ]
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0 
                    )
                )
                in_channels = out_channels
                
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
                
        return layers
    
class Yolov3(nn.Module):
    def __init__(self,device,anchors, config = None, num_classes=10):
        super().__init__()
        self.backbone = Yolov3LoadPretrain(in_channels=1)
        
        self.config = config if config is not None else []
        self.in_channels = 1024
        self.num_classes = num_classes
        self.layers = self._create_conv_layers()
        self.device = device
        self.anchors = anchors
                
    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.backbone.children():  # Duyệt qua các mô-đun con
            for sub_layer in layer:
                if isinstance(sub_layer, ResidualBlock) and sub_layer.num_repeats == 8:   
                    route_connections.append(sub_layer(x))
                x = sub_layer(x)
        
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue 
            
            x = layer(x)
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
                
        return outputs
    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels 

        for module in self.config:
            if isinstance(module, tuple):  # Thêm các CNNBlock dựa trên tuple
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels
            
            elif isinstance(module, str):  # "S" hoặc "U" từ config
                if module == "S":  # ScalePrediction
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                    
                elif module == "U":  # Upsample
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3  # Ghép nối với đặc trưng trước đó (concat)
        
        return layers
        
    def load_pretrained_weights(self, pretrained_path):
        device = next(self.parameters()).device  
        pretrained_weights = torch.load(pretrained_path, map_location=device)

        filtered_weights = {
            k: v for k, v in pretrained_weights.items() if "fc" not in k
        }

        self.backbone.load_state_dict(filtered_weights, strict=False)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def decode_predictions(self,image,
                           iou_threshold=0.5, score_threshold=0.5):
        if not isinstance(image,torch.Tensor):
            image = np.squeeze(image)
            image = torch.tensor(image)
            
        image = image.view(1,1,image.size(0),image.size(1)).to(self.device)
        image_size = image.size(2)
        print(image.dtype)
        outputs = self(image)
        batch_size = outputs[0].size(0)
        num_anchors_per_scale = outputs[0].size(1)
        num_scales = len(outputs)
        grid_sizes = [output.size(2) for output in outputs]
    
        # Concatenate outputs from all scales
        outputs = torch.cat([output.reshape(batch_size, -1, self.num_classes + 5) for output in outputs], dim=1)
    
        obj_pred = outputs[..., 4:5]  
        class_pred = outputs[..., 5:] 
        box_pred = outputs[..., :4]  
        
        anchors = torch.tensor(self.anchors).to(self.device).view(-1, 2)
        grid_offsets = []
        anchor_list = []
        cell_pixels = []
        for idx, grid_size in enumerate(grid_sizes):
            num_cells = grid_size * grid_size
            cell_size = image_size / grid_size
            cell_size = torch.tensor(cell_size).view(1, 1, 1).repeat(batch_size, num_anchors_per_scale * num_cells, 1)
            cell_pixels.append(cell_size.to(self.device))
    
            # Map anchors to grid cells
            current_anchors = anchors[idx * num_anchors_per_scale: num_anchors_per_scale * (idx + 1)]
            current_anchors = current_anchors.view(1, num_anchors_per_scale, 2)
            current_anchors = current_anchors.repeat(batch_size, num_cells, 1)
            anchor_list.append(current_anchors)
    
            # Grid offsets
            grid_x = torch.arange(grid_size).repeat(grid_size, 1).view(-1)
            grid_y = torch.arange(grid_size).repeat_interleave(grid_size).view(-1)
            grid_coords = torch.stack([grid_x, grid_y], dim=1).float().to(anchors.device)
    
            grid_coords = grid_coords.repeat(num_anchors_per_scale, 1)
            grid_coords = grid_coords.view(1, -1, 2).repeat(batch_size, 1, 1)
            grid_offsets.append(grid_coords)
    
        anchors = torch.cat(anchor_list, dim=1)
        grid_offsets = torch.cat(grid_offsets, dim=1)
        cell_pixels = torch.cat(cell_pixels, dim=1)
    
        # Decode the bounding boxes
        tx, ty, tw, th = box_pred[..., 0], box_pred[..., 1], box_pred[..., 2], box_pred[..., 3]
        bx = (torch.sigmoid(tx) + grid_offsets[..., 0]) * cell_pixels[..., 0]
        by = (torch.sigmoid(ty) + grid_offsets[..., 1]) * cell_pixels[..., 0]
        bw = torch.exp(tw) * anchors[..., 0] * cell_pixels[..., 0]
        bh = torch.exp(th) * anchors[..., 1] * cell_pixels[..., 0]
    
        x_min = torch.clamp(bx - bw / 2,min = 0, max = image_size)
        y_min = torch.clamp(by - bh / 2,min = 0, max = image_size)
        x_max = torch.clamp(bx + bw / 2,min = 0, max = image_size)
        y_max = torch.clamp(by + bh / 2,min = 0, max = image_size)
    
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    
        # Normalize predictions
        obj_pred = torch.sigmoid(obj_pred)
        class_pred = F.softmax(class_pred, dim=-1)
        class_probs, class_labels = class_pred.max(dim=-1)
        combinate_pred = class_probs.unsqueeze(-1) * obj_pred
        # Filter by score threshold
        scores = combinate_pred.squeeze(-1)
        mask = scores > score_threshold
        # boxes batch x num_box 
        boxes = boxes[mask]
        
        scores = scores[mask]
        class_labels = class_labels[mask]
    
        # Apply Non-Maximum Suppression (NMS)
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_labels = class_labels[keep_indices]
    
        return boxes, scores, class_labels
    
    def tunningInput(self, img,iou_threshold=0.5, score_threshold=0.5):
        image_array = np.array(img)/255.0
        image_array = torch.tensor(image_array, dtype=torch.float32)
        outbox = self.decode_predictions(image_array,score_threshold = score_threshold)
        return outbox

    