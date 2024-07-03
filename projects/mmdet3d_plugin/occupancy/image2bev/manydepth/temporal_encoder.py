 
import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from collections import OrderedDict

import torch
from torchvision import transforms
import sys
sys.path.append("projects/mmdet3d_plugin/occupancy/image2bev/manydepth/")
import networks
from torch.autograd import Variable



class temporal_encoder(torch.nn.Module):
    def __init__(self, maxdisp, width, height  ):
        super(temporal_encoder, self).__init__()
        self.maxdisp = maxdisp
        
        self.pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        self.pose_dec = networks.PoseDecoder(self.pose_enc.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)
        self.encoder = networks.ResnetEncoderMatching(18, False,
                                                input_width = width, 
                                                input_height = height, 
                                                adaptive_bins=True,
                                                num_depth_bins=maxdisp )
        for name, p in self.named_parameters():
            if name.startswith("pose"):
                p.requires_grad = False


    def load_and_preprocess_image(self, image ):
      
        batch, channel, original_width, original_height = image.shape
        image = image.div(255)
        image = Variable(image, requires_grad=True).cuda() 
        if torch.cuda.is_available():
            return image, (original_height, original_width)
        return image, (original_height, original_width)

    def load_and_preprocess_intrinsics(self, intrinsics, resize_width, resize_height):
        K = np.array( intrinsics )
        K[0, :] *= resize_width // 4
        K[1, :] *= resize_height // 4
        invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
        invK = Variable(invK, requires_grad=True).cuda() 
        K = torch.Tensor(K).unsqueeze(0)
        K = Variable(K, requires_grad=True).cuda() 
        return K , invK 


    def forward(self, ref_images, source_images, intrinsics, calib=None ):
        B, T, C, H, W = source_images.shape 
        combined_waped_feature = torch.zeros( B, T, self.maxdisp, H//4, W//4 ).cuda()

        height, width = ref_images.shape[-2: ]
        intrinsics =  intrinsics.squeeze(1).cpu().detach().numpy()  
        
        K, invK = torch.zeros_like( torch.tensor(intrinsics)).cuda() , torch.zeros_like( torch.tensor(intrinsics)).cuda()
        for batch in range( 0, B ):
            K_, invK_ = self.load_and_preprocess_intrinsics(intrinsics[batch], width, height)
            K_ = Variable(K_, requires_grad=True).cuda()
            invK_ = Variable(invK_, requires_grad=True).cuda()

            K[batch], invK[batch] = K_, invK_

        ref_image = ref_images.squeeze(1)
        for temporal in range( 0, T ):
            source_image = source_images[:, temporal, ...]
            input_image, original_size = self.load_and_preprocess_image(ref_image )
            source_image, _ = self.load_and_preprocess_image(source_image )

            with torch.no_grad():
            # Estimate poses
                pose_inputs = [source_image, input_image]
                pose_inputs = self.pose_enc(torch.cat(pose_inputs, 1))
                pose_inputs = [ pose_inputs ]
                axisangle, translation = self.pose_dec(pose_inputs)
                pose = networks.transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

        
            curr_feature, batch_waped_feature  = self.encoder(current_image=input_image,
                                            lookup_images=source_image.unsqueeze(1),
                                            poses=pose.unsqueeze(1),
                                            K=K,
                                            invK=invK)
            combined_waped_feature[:, temporal,:,:,:] = batch_waped_feature.squeeze(1)
        
        return  curr_feature, combined_waped_feature


