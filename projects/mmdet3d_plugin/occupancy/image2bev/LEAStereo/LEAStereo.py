
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os,sys
import skimage
import skimage.io
import skimage.transform
from PIL import Image

import argparse
sys.path.append('projects/mmdet3d_plugin/occupancy/image2bev/LEAStereo/')
from models.decoding_formulas import network_layer_to_space
from build_model_2d import Disp
from new_model_2d import newFeature
from skip_model_3d import newMatching
from collections import OrderedDict


def warp( x, calib, down, maxdepth ):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, D, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    # B,C,D,H,W to B,H,W,C,D
    x = x.transpose(1, 3).transpose(2, 4)
    B, H, W, C, D = x.size()
    x = x.view(B, -1, C, D)
    # mesh grid
    xx = (calib / ( down * 4.))[:, None] / torch.arange(1, 1 +  maxdepth //  down,
                                                            device='cuda').float()[None, :]
    new_D =  maxdepth //  down
    xx = xx.view(B, 1, new_D).repeat(1, C, 1)
    xx = xx.view(B, C, new_D, 1)
    yy = torch.arange(0, C, device='cuda').view(-1, 1).repeat(1, new_D).float()
    yy = yy.view(1, C, new_D, 1).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), -1).float()
    vgrid = Variable(grid)
 
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(D - 1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(C - 1, 1) - 1.0
    if float(torch.__version__[:3])>1.2:
        output = nn.functional.grid_sample(x, vgrid, align_corners=True).contiguous()
    else:
        output = nn.functional.grid_sample(x, vgrid).contiguous()
    output = output.view(B, H, W, C, new_D).transpose(1, 3).transpose(2, 4)
    return output.contiguous()

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume



class LEA_encoder(nn.Module):
    def __init__(self, maxdisp=192):
        super(LEA_encoder, self).__init__()
        self.opt = None

        network_path_fea, cell_arch_fea = np.array([1, 0, 1, 0, 0, 0]), np.array([[0, 1],[1, 0], [3, 1], [4, 1], [8, 1], [5, 1]])
        network_path_mat, cell_arch_mat = np.array([1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 0, 1]), np.array([[1, 1], [0, 1], [3, 1], [4, 1],[8, 1],[6, 1]])


        network_arch_fea = network_layer_to_space(network_path_fea)
        network_arch_mat = network_layer_to_space(network_path_mat)

        self.maxdisp = maxdisp
        self.feature = newFeature(network_arch_fea, cell_arch_fea, args=self.opt)  
        self.matching= newMatching(network_arch_mat, cell_arch_mat, args=self.opt) 
        self.disp = Disp(self.maxdisp)
     

    def forward(self, x, y, calib):  
       
        x = self.feature(x)     
        y  = self.feature(y)    

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.maxdisp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
        cost = warp(cost , calib, down=1, maxdepth=cost.shape[2] )

        initial_volume = cost
        cost = self.matching(cost)   
        classfy_volume = cost

        disp = self.disp(cost)  

        return { "initial_volume": initial_volume, "classfy_volume": classfy_volume, disp:"disp" } 
