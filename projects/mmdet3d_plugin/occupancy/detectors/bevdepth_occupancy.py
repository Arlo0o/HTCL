import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import mmcv
import collections 

from mmdet.models import DETECTORS
from mmdet3d.models import builder, losses
from collections import OrderedDict
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict
from sklearn.metrics import confusion_matrix as CM
from .bevdepth import BEVDepth, BEVDepth4D
from projects.mmdet3d_plugin.utils import fast_hist_crop
from projects.mmdet3d_plugin.models.utils import GridMask

import numpy as np
import time
import pdb

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
class FuseNet(nn.Module):
    def __init__(self, in_channel):
        super(FuseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=in_channel, out_channels=16, kernel_size=3, stride=1, pad=1)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=in_channel, kernel_size=3, stride=1, pad=1)
        self.conv2 = nn.Conv3d(in_channels=in_channel, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1= self.conv0(x)
        x2= self.conv1(x1)
        # x3 = x*self.sigmoid(x2)
        out = self.conv2(x2) 
        return out
        
@DETECTORS.register_module()
class BEVDepthOccupancy(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            use_grid_mask=False,
            disable_loss_depth=False,
            queue_length=None,
            **kwargs):
        super().__init__(**kwargs)
  
        # if queue_length>1:
        #     self.FuseNet = FuseNet(in_channel=3)
        
        self.loss_cfg = loss_cfg
        self.use_grid_mask = use_grid_mask
        self.disable_loss_depth = disable_loss_depth
        
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)
        
        if self.use_grid_mask:
            imgs = self.grid_mask(imgs)
        
        x = self.img_backbone(imgs) 

        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    

    def image_encoder_source(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)
        
        if self.use_grid_mask:
            imgs = self.grid_mask(imgs)
        
        x = self.img_backbone(imgs) 

        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x

    @force_fp32()
    def bev_encoder(self, x, depth, temporal_voxel):  
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        x = self.img_bev_encoder_backbone(x)  
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['bev_encoder'].append(t1 - t0)
        
        x = self.img_bev_encoder_neck(x, depth ,temporal_voxel)  
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['bev_neck'].append(t2 - t1)
        
        return x
    
    def extract_img_feat(self, img, img_metas, gt, mode):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

 
        img_left, img_right = img[0][0], img[1][0]  ### B Temporal N C H W
        B, T, N, C, H, W = img_left.shape

        img_left_ref, img_right_ref = img_left[ :, -1, ... ], img_right[ :, -1, ... ]
        img_left_sour, img_right_sour = img_left[:,:-1, ... ].squeeze(2).contiguous(), img_right[:,:-1, ... ].squeeze(2).contiguous()

        img_left_ref_feature = self.image_encoder( img_left_ref ) ### B Temporal  C H W
        # img_left_sour_feature = self.image_encoder_source( img_left_sour ) ### B Temporal  C H W



        # if T>1:
        #     sourcel_fuse = self.FuseNet(sourcel)
        #     sourcer_fuse = self.FuseNet(sourcer)


        x, x2 = img_left_ref_feature, None 
        img_feats = x.clone()
         
        img, img2 = img[0], img[1]
        filenamesl, filenamesr = img[-1], img2[-1]  ### [2, 4, 1280, 384, 3]


        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]  
        rots2, trans2, intrins2, post_rots2, post_trans2, bda2 = img2[1:7]
        
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)   
        mlp_input2 = self.img_view_transformer.get_mlp_input(rots2, trans2, intrins2, post_rots2, post_trans2, bda2)  
        
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]   
        geo_inputs2 = [rots2, trans2, intrins2, post_rots2, post_trans2, bda2, mlp_input2]   

        calib = img[9]
        
        x, depth, temporal_voxel = self.img_view_transformer([x] + geo_inputs + [x2] + geo_inputs2 + [calib]+ [img, img2], gt, mode, img[0], img2[0], filenamesl, filenamesr)  


        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        x = self.bev_encoder(x, depth, temporal_voxel)
        if type(x) is not list:
            x = [x]
        
        return x, depth, img_feats, temporal_voxel

    def extract_feat(self, points, img, img_metas, gt, mode):
        """Extract features from images and points."""
        
        voxel_feats, depth, img_feats, temporal_voxel = self.extract_img_feat(img, img_metas, gt, mode)
        pts_feats = None
        return (voxel_feats, img_feats, depth, temporal_voxel )
    
    @force_fp32(apply_to=('pts_feats'))
    def forward_pts_train(
            self,
            pts_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            img_feats=None,
            points_uv=None,
            **kwargs,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        outs = self.pts_bbox_head(
            voxel_feats=pts_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            points_uv=points_uv,
            **kwargs,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_head'].append(t1 - t0)
        
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            target_voxels=gt_occ,
            output_points=outs['output_points'],
            target_points=points_occ,
            img_metas=img_metas,
            **kwargs,
        )    ## gt_occ 1, 256, 256, 32   points_occ[20120, 4]
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['loss_occ'].append(t2 - t1)
        
        return losses


    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            points_uv=None,
            **kwargs,
        ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # extract bird-eye-view features from perspective images
        voxel_feats, img_feats, depth, temporal_voxel = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, gt=img_inputs[0][7].clone(), mode='train')
        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        if not self.disable_loss_depth: ## True
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[0][7].clone(), depth)  


        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
            
        if self.img_bev_encoder_backbone.crp3d:
            losses['loss_rel_ce'] = self.img_bev_encoder_backbone.crp_loss(
                CP_mega_matrices=kwargs['CP_mega_matrix'],
            )
        
        if self.img_view_transformer.imgseg:
            losses['loss_imgseg'] = self.img_view_transformer.get_seg_loss(
                seg_labels=kwargs['img_seg'],
            )
        

        ## voxel_feats [4, 384, 128, 128, 16]  gt_occ[4, 256, 256, 32]
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ, 
                        points_occ, img_metas, img_feats=img_feats, points_uv=points_uv, **kwargs)  \
        
        losses_occupancy2 = self.forward_pts_train(temporal_voxel, gt_occ,  points_occ, img_metas, img_feats=img_feats, points_uv=points_uv, **kwargs)
        for key,value in losses_occupancy2.items():  losses_occupancy[key] += value

        losses.update(losses_occupancy)
        
        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            print(out_res)
        
        if self.record_time:
            logging_latencies()
        
        return losses
        
    def forward_test(self,
            img_metas=None,
            img_inputs=None,
            **kwargs,
        ):
        
        return self.simple_test(img_metas, img_inputs, **kwargs)
    
    def simple_test(self, img_metas, img=None, rescale=False, points_occ=None, gt_occ=None, points_uv=None):
        
        voxel_feats, img_feats, depth, temporal_voxel = self.extract_feat(points=None, img=img, img_metas=img_metas, gt=None, mode='val')        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            points_uv=points_uv,
        )
        
        # evaluate nusc lidar-seg
        if output['output_points'] is not None and points_occ is not None:
            output['evaluation_semantic'] = self.simple_evaluation_semantic(output['output_points'], points_occ, img_metas)
        else:
            output['evaluation_semantic'] = 0
            
        # evaluate voxel 
        output['output_voxels'] = F.interpolate(output['output_voxels'][0], 
                    size=gt_occ.shape[1:], mode='trilinear', align_corners=False)
        output['target_voxels'] = gt_occ
        
        output['target_depth'] = img[0][7].clone()
        output['output_depth'] = depth
        
        
        return output
    
    def post_process_semantic(self, pred_occ):
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
        score, color = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        
        return color

    def simple_evaluation_semantic(self, pred, gt, img_metas):
        pred = torch.argmax(pred[0], dim=1).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt[:, 3].astype(np.int)
        unique_label = np.arange(16)
        
        hist = fast_hist_crop(pred, gt, unique_label)
        
        return hist
    
    def evaluation_semantic(self, pred, gt, img_metas):
        import open3d as o3d

        assert pred.shape[0] == 1
        pred = pred[0]
        gt_ = gt[0].cpu().numpy()
        
        x = np.linspace(0, pred.shape[0] - 1, pred.shape[0])
        y = np.linspace(0, pred.shape[1] - 1, pred.shape[1])
        z = np.linspace(0, pred.shape[2] - 1, pred.shape[2])
    
        X, Y, Z = np.meshgrid(x, y, z,  indexing='ij')
        vv = np.stack([X, Y, Z], axis=-1)
        pred_fore_mask = pred > 0
        
        if pred_fore_mask.sum() == 0:
            return None
        
        # select foreground 3d voxel vertex
        vv = vv[pred_fore_mask]
        vv[:, 0] = (vv[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
        vv[:, 1] = (vv[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
        vv[:, 2] = (vv[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vv)
        
        # for every lidar point, search its nearest *foreground* voxel vertex as the semantic prediction
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        indices = []
        
        for vert in gt_[:, :3]:
            _, inds, _ = kdtree.search_knn_vector_3d(vert, 1)
            indices.append(inds[0])
        
        gt_valid = gt_[:, 3].astype(np.int)
        pred_valid = pred[pred_fore_mask][np.array(indices)]
        
        mask = gt_valid > 0
        cm = CM(gt_valid[mask] - 1, pred_valid[mask] - 1, labels=np.arange(16))
        cm = cm.astype(np.float32)
        
        return cm


    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, bda):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)  
        if intrins.shape[3] == 4: # for KITTI
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points


    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        if self.use_bev_pool:
            final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0],
                                   self.nx[1])
            final = final.transpose(dim0=-2, dim1=-1)
        else:
            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
            # cumsum trick
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
            # griddify (B x C x Z x X x Y)
            final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        return final


@DETECTORS.register_module()
class BEVDepthOccupancy4D(BEVDepthOccupancy):
    def prepare_voxel_feat(self, img, rot, tran, intrin, 
                post_rot, post_tran, bda, mlp_input):
        
        x = self.image_encoder(img)
        img_feats = x.clone()
        
        voxel_feat, depth = self.img_view_transformer([x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        
        return voxel_feat, depth, img_feats

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        voxel_feat_list = []
        img_feat_list = []
        depth_list = []
        key_frame = True # back propagation for key frame only
        
        for img, rot, tran, intrin, post_rot, \
            post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
                
            mlp_input = self.img_view_transformer.get_mlp_input(
                rots[0], trans[0], intrin,post_rot, post_tran, bda)
            inputs_curr = (img, rot, tran, intrin, post_rot, post_tran, bda, mlp_input)
            if not key_frame:
                with torch.no_grad():
                    voxel_feat, depth, img_feats = self.prepare_voxel_feat(*inputs_curr)
            else:
                voxel_feat, depth, img_feats = self.prepare_voxel_feat(*inputs_curr)
            
            voxel_feat_list.append(voxel_feat)
            img_feat_list.append(img_feats)
            depth_list.append(depth)
            key_frame = False
        
        voxel_feat = torch.cat(voxel_feat_list, dim=1)
        x = self.bev_encoder(voxel_feat)
        if type(x) is not list:
            x = [x]

        return x, depth_list[0], img_feat_list[0]
        