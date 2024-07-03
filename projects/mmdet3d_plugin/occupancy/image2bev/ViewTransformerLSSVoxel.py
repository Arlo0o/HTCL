# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool import bev_pool
from mmdet3d.ops.voxel_pooling import voxel_pooling
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
from projects.mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from mmdet.models.backbones.resnet import BasicBlock
from projects.mmdet3d_plugin.utils.semkitti import semantic_kitti_class_frequencies, kitti_class_names, CE_ssc_loss
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from PIL import Image
import pdb
import sys
from .ViewTransformerLSSBEVDepth import *
from .semkitti_depthnet import SemKITTIDepthNet
from .temporal_retrieve  import *
norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)
from gwc_encoder import * 
sys.path.append('projects/mmdet3d_plugin/occupancy/image2bev/')
from LEAStereo.LEAStereo import LEA_encoder
from  manydepth.temporal_encoder import  temporal_encoder

 
class volume_interaction(nn.Module):  
    def __init__(self,  out_channels=1):
        super(volume_interaction, self).__init__()
        self.dres1 = nn.Sequential(convbn_3d(2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   hourglass(32))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.out3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, 1, 3, 1, 1))
    def forward(self, stereo_volume, lss_volume): 
        stereo_volume=stereo_volume.unsqueeze(1)
        lss_volume=lss_volume.unsqueeze(1)
        all_volume=torch.cat( (stereo_volume, lss_volume ), dim=1)
        data1_ = self.dres1(all_volume)
        data2_ = self.dres2(data1_)
        data3 = self.dres3(data2_) + data1_
        data3 =  self.out3(data3) 
        data3 = data3.squeeze(1) 
        data3 = F.softmax(data3, dim=1)
        data1, data2=None, None
        return data3, [data1, data2]


class temporal_interaction(nn.Module):  
    def __init__(self,  out_channels=1):
        super(volume_interaction, self).__init__()
        self.dres1 = nn.Sequential(convbn_3d(2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   hourglass(32))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.out3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, 1, 3, 1, 1))
    def forward(self, stereo_volume, lss_volume):  
        stereo_volume=stereo_volume.unsqueeze(1)
        lss_volume=lss_volume.unsqueeze(1)
        all_volume=torch.cat( (stereo_volume, lss_volume ), dim=1)
        data1_ = self.dres1(all_volume)
        data2_ = self.dres2(data1_)
        data3 = self.dres3(data2_) + data1_
        data3 =  self.out3(data3) 
        data3 = data3.squeeze(1) 
        data3 = F.softmax(data3, dim=1)
        data1, data2=None, None
        return data3, [data1, data2]

    
    
@NECKS.register_module()
class ViewTransformerLiftSplatShootVoxel(ViewTransformerLSSBEVDepth):
    def __init__(
            self, 
            loss_depth_weight,
            semkitti=False,
            imgseg=False,
            imgseg_class=20,
            lift_with_imgseg=False,
            point_cloud_range=None,
            loss_seg_weight=1.0,
            loss_depth_type='bce', ##'bce', smooth
            point_xyz_channel=0,
            point_xyz_mode='cat',
            depth_model="lea",
            temporal_num = 4,
            **kwargs,
        ):
        
        super(ViewTransformerLiftSplatShootVoxel, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)
    
        self.leamodel=LEA_encoder(maxdisp=192) 
        self.volume_interaction = volume_interaction()
        self.temporal_deformable = multipatch_deformable( indim=3, outdim=3 )
        self.curr_patch = multi_patch2d(in_channel=64, depth=1)
        self.warped_patch = multi_patch3d(in_channel=3, depth=1)
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.temporal_encoder = temporal_encoder( maxdisp=112, height=384, width=1280 )
        self.temporal_prehourglass = nn.Sequential(convbn_3d( temporal_num-1, 32, 3, 1, 1),  nn.ReLU(inplace=True),
                                                hourglass(32), 
                                                convbn_3d(32, 64, 3, 1, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, bias=False))

        self.temporal_hourglass = nn.Sequential(convbn_3d( 64, 32, 3, 1, 1),  nn.ReLU(inplace=True),
                                                hourglass(32), 
                                                convbn_3d(32, 64, 3, 1, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(64, 384, kernel_size=3, padding=1, stride=1, bias=False))
        


        self.semkitti = semkitti

        self.loss_depth_type = loss_depth_type
        self.cam_depth_range = self.grid_config['dbound']
        self.constant_std = 0.5
        self.point_cloud_range = point_cloud_range
        
        ''' Extra input for Splating: except for the image features, the lifted points should also contain their positional information '''
        self.point_xyz_mode = point_xyz_mode
        self.point_xyz_channel = point_xyz_channel
        
        assert self.point_xyz_mode in ['cat', 'add']
        if self.point_xyz_mode == 'add':
            self.point_xyz_channel = self.numC_Trans
        
        if self.point_xyz_channel > 0:
            assert self.point_cloud_range is not None
            self.point_cloud_range = torch.tensor(self.point_cloud_range)
            
            mid_channel = self.point_xyz_channel // 2
            self.point_xyz_encoder = nn.Sequential(
                nn.Linear(in_features=3, out_features=mid_channel),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=mid_channel, out_features=self.point_xyz_channel),
            )
    
            
        ''' Auxiliary task: image-view segmentation '''
        self.imgseg = imgseg
        if self.imgseg:
            self.imgseg_class = imgseg_class
            self.loss_seg_weight = loss_seg_weight
            self.lift_with_imgseg = lift_with_imgseg
            
            # build a small segmentation head
            in_channels = self.numC_input
            self.img_seg_head = nn.Sequential(
                BasicBlock(in_channels, in_channels),
                BasicBlock(in_channels, in_channels),
                nn.Conv2d(in_channels, self.imgseg_class, kernel_size=1, padding=0),
            )
        
        self.forward_dic = {}


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape  ## [1, 1, 384, 1280]
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)  
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample) #
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths) 
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  #
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))  
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:] 
        
        return gt_depths_vals, gt_depths.float()
    
    def get_diff_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape  
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)   
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample) 
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)  
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values   
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)  
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]  
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))  
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1)[:, :, :, 1:]

        mask = torch.max(gt_depths, dim=3).values > 0.0
        mask = mask.unsqueeze(3)
        gt_depths = gt_depths * mask
        
        gt_depths = gt_depths.permute(0, 3, 1, 2).contiguous()
        gt_depths = gt_depths.unsqueeze(1)  

        
        return gt_depths.float()
    
    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):  
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels) 
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)  
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        return depth_loss

    @force_fp32()
    def get_smooth_depth_loss(self, depth_labels, depth_preds):  
       
        B,D,H,W = depth_preds.shape
        depth_labels =  F.interpolate(depth_labels, [ H, W], mode='bilinear', align_corners=False)  

        with torch.cuda.device_of(depth_preds):
            disp = torch.reshape(torch.arange(0, D, device=torch.cuda.current_device(), dtype=torch.float32),[1,D,1,1])
            disp = disp.repeat(depth_preds.size()[0], 1, depth_preds.size()[2], depth_preds.size()[3])
            depth_preds = torch.sum(depth_preds * disp, 1).unsqueeze(1) 

        mask = (depth_labels > 0)  
        mask.detach_()
        loss = F.smooth_l1_loss(depth_preds[mask], depth_labels[mask], reduction='mean')
        return loss


    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.cam_depth_range, constant_std=self.constant_std)
        
        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))        
        
        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]
        
        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)
        
        return depth_loss
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)
        
        elif self.loss_depth_type == 'kld':
            depth_loss = self.get_klv_depth_loss(depth_labels, depth_preds)
        
        elif self.loss_depth_type == 'smooth':
            depth_loss = self.get_smooth_depth_loss(depth_labels, depth_preds)
        
        else:
            pdb.set_trace()
        
        return self.loss_depth_weight * depth_loss

    @force_fp32()
    def get_seg_loss(self, seg_labels):
        class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001)).type_as(seg_labels).float()
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=0, reduction="mean",
        )
        seg_preds = self.forward_dic['imgseg_logits']
        if seg_preds.shape[-2:] != seg_labels.shape[-2:]:
            seg_preds = F.interpolate(seg_preds, size=seg_labels.shape[1:])
        
        loss_seg = criterion(seg_preds, seg_labels.long())
        
        return self.loss_seg_weight * loss_seg
        
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape  
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)  
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_xyz = geom_feats.clone()
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        if self.point_xyz_channel > 0:
            geom_xyz = geom_xyz.view(Nprime, 3)
            geom_xyz = geom_xyz[kept]
            
            pc_range = self.point_cloud_range.type_as(geom_xyz) # normalize points to [-1, 1]
            geom_xyz = (geom_xyz - pc_range[:3]) / (pc_range[3:] - pc_range[:3])
            geom_xyz = (geom_xyz - 0.5) * 2
            geom_xyz_feats = self.point_xyz_encoder(geom_xyz)
            
            if self.point_xyz_mode == 'cat':
                # concatenate image features & geometric features
                x = torch.cat((x, geom_xyz_feats), dim=1)
            
            elif self.point_xyz_mode == 'add':
                x += geom_xyz_feats
                
            else:
                raise NotImplementedError

        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)

        return final

    def forward(self, input, gt, mode, imgl, imgr, left_input, right_input  ):
        
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        
        B, N, C, H, W = x.shape 
        x = x.view(B * N, C, H, W)

        calib = input[16]

        if  imgl.shape[1]>1:
            imgl, imgr = imgl[:, -1, ...], imgr[:, -1, ...]
        imgl, imgr = F.interpolate(imgl.squeeze(1), size=[288, 960], mode='bilinear', align_corners=True), F.interpolate(imgr.squeeze(1), size=[288, 960], mode='bilinear', align_corners=True) 
        stereo_volume = self.leamodel(imgl, imgr, calib )["classfy_volume"]   
        stereo_volume = F.interpolate(stereo_volume, size=[ 112, H, W ], mode='trilinear', align_corners=True).squeeze(1)  
        stereo_volume = F.softmax(-stereo_volume, dim=1)


        if self.imgseg:  
            self.forward_dic['imgseg_logits'] = self.img_seg_head(x)
        
        x = self.depth_net(x, mlp_input)  
        depth_digit = x[:, :self.D, ...]  
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...] 
        depth_prob = self.get_depth_dist(depth_digit) 
 
        depth_prob, auxility = self.volume_interaction(stereo_volume, depth_prob)  
       
        
        if self.imgseg and self.lift_with_imgseg:
            img_segprob = torch.softmax(self.forward_dic['imgseg_logits'], dim=1)
            img_feat = torch.cat((img_feat, img_segprob), dim=1)

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)   
        volume = volume.view(B, N, -1, self.D, H, W)  
        volume = volume.permute(0, 1, 3, 4, 5, 2)  
 
        # Splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)  
        bev_feat = self.voxel_pooling(geom, volume) 
        

        img_left_ref, img_left_sour = left_input[:, -1, ...].unsqueeze(1).permute(0,1,4,2,3).cuda(), left_input[:,:-1, ...].permute(0,1,4,2,3).cuda()  #
        curr_feature, batch_waped_feature = self.temporal_encoder( ref_images=img_left_ref, source_images=img_left_sour, intrinsics=intrins ) #
        

        curr_feature = F.interpolate(curr_feature, size=[H, W], mode='bilinear', align_corners=True) 
        batch_waped_feature = F.interpolate(batch_waped_feature, size=[self.D, H, W], mode='trilinear', align_corners=True) 

        defomable_batch_waped_feature = self.temporal_deformable( batch_waped_feature )  
        
        curr_feature = self.curr_patch( curr_feature ) 
        batch_waped_feature = self.warped_patch( batch_waped_feature ) 
 

        temporal_volume = self.cossim(  (curr_feature-curr_feature.mean(1).unsqueeze(1)).unsqueeze(2).repeat(1,1,self.D,1,1), (batch_waped_feature-batch_waped_feature.mean(1).unsqueeze(1)) ).unsqueeze(1)
        temporal_volume = (temporal_volume) * defomable_batch_waped_feature
        
        temporal_volume = self.temporal_prehourglass( temporal_volume )  
          
        temporal_volume = temporal_volume.view(B, N, -1, self.D, H, W)  
        temporal_volume = temporal_volume.permute(0, 1, 3, 4, 5, 2) 
        temporal_volume = self.voxel_pooling(geom, temporal_volume)   
        temporal_volume =  self.temporal_hourglass(temporal_volume ) 
        temporal_voxel = [temporal_volume]


        return bev_feat, depth_prob, temporal_voxel