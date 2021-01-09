"""
    define model here
"""

import torch.nn as nn
import torchvision.transforms as transforms
from .resnet import resnet50
import torch.nn.functional as F
from .helper import *
import torch
import numpy as np

class EastModel(nn.Module):

    def __init__(self, text_scale = 512, input_size = 512):
        super(EastModel, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.text_scale = text_scale
        self.input_size = input_size
        #channel = [None, 512, 256, 128]
        channel = [None, 128, 64, 32]
        stage_out_channels = [2048, 1024, 512, 256]
        stage_out_sizes = [input_size // 32, input_size // 16, input_size // 8, input_size // 4]
        self.h_operations = [None] * 4
        self.g_operations = [None] * 4
        for i in range(4):
            if i == 1:
                self.h_operations[i] = nn.Sequential(
                        nn.Conv2d(stage_out_channels[i]+stage_out_channels[i-1], channel[i],
                            kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(channel[i]),
                        nn.ReLU(),
                        nn.Conv2d(channel[i], channel[i],
                            kernel_size=3, stride=1, bias=False, padding=1),
                        nn.BatchNorm2d(channel[i]),
                        nn.ReLU()
                    )
            elif i > 1:
                self.h_operations[i] = nn.Sequential(
                        nn.Conv2d(stage_out_channels[i]+channel[i-1], channel[i],
                            kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(channel[i]),
                        nn.ReLU(),
                        nn.Conv2d(channel[i], channel[i],
                            kernel_size=3, stride=1, bias=False, padding=1),
                        nn.BatchNorm2d(channel[i]),
                        nn.ReLU()
                    )
            if i < 3:
                self.g_operations[i] = nn.Upsample(size=[stage_out_sizes[i+1],stage_out_sizes[i+1]], mode='bilinear')
            else:
                self.g_operations[i] = nn.Sequential(
                        nn.Conv2d(channel[i], channel[i], kernel_size=3, stride=1, bias=False, padding=1),
                        nn.BatchNorm2d(channel[i]),
                        nn.ReLU()
                )
        # This step allows pyotrch put list to cuda
        self.h_operations = nn.ModuleList(self.h_operations)
        self.g_operations = nn.ModuleList(self.g_operations)
        self.score_operations = nn.Sequential(
                                    nn.Conv2d(channel[-1], 1,
                                        kernel_size=1, stride=1, bias=False),
                                    nn.Sigmoid()
                                )
        self.geo_operations = nn.Sequential(
                                    nn.Conv2d(channel[-1], 4,
                                        kernel_size=1, stride=1, bias=False),
                                    nn.Sigmoid()
                                )
        self.angle_operations = nn.Sequential(
                                    nn.Conv2d(channel[-1], 1,
                                        kernel_size=1, stride=1, bias=False),
                                    nn.Sigmoid()
                                )

    def forward(self, images):
        images = mean_image_subtraction(images)
        _ = self.resnet50(images)
        f = [self.resnet50.stage4, self.resnet50.stage3, \
                self.resnet50.stage2, self.resnet50.stage1]
        g = [None, None, None, None]
        h = [None, None, None, None]
        
        for i in range(4):
            if i == 0:
                h[i] = f[i]
            else:
                tmp = torch.cat([f[i],g[i-1]],1)
                h[i] = self.h_operations[i](tmp)
            g[i] = self.g_operations[i](h[i])

        F_score = self.score_operations(g[3])

        # 4 channel of axis aligned bbox and 1 channel rotation angle
        geo_map = self.geo_operations(g[3]) * self.text_scale
        angle_map = (self.angle_operations(g[3]) - 0.5) * np.pi/2 # angle is between [-45, 45]
        F_geometry = torch.cat([geo_map, angle_map], 1)

        return F_score, F_geometry

    

