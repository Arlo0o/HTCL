from .swinunetr import SwinUNETR
from mmdet3d.models.builder import BACKBONES
from mmcv.runner import BaseModule
import torch
from collections import OrderedDict
import torch.nn.functional as F
from .crp3d import CPMegaVoxels

@BACKBONES.register_module()
class Swintransformer3D(BaseModule):
    def __init__(self, crp3d=False ):
        super().__init__()

        self.crp3d = crp3d
        self.model = SwinUNETR(
            # img_size=(256, 256, 32),
            img_size=(128, 128, 32),
            in_channels=128,
            out_channels=384,
            feature_size=48,
            use_checkpoint=True,
            )
        # checkpointpath = "/code/occupancy-lss/occupancy-lss-0603/fold1_f48_ep300_4gpu_dice0_9059/model.pt"
        # pretrained_dict = torch.load(checkpointpath , map_location='cpu')['state_dict']
        # new_state_dict = OrderedDict()
        # for k, v in pretrained_dict.items():
        #     if not (k.startswith( "out.conv.conv" ) or k.startswith( "encoder1.layer") or k.startswith( "swinViT.patch_embed.proj") ) :
        #         name = k 
        #         new_state_dict[name] = v #新字典的key值对应的value一一对应
        # model_dict= self.model.state_dict()
        # pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
    
        # for k,v in pretrained_dict.items() : print("------------", k)
        # self.model.load_state_dict(model_dict)
        # print("reload model!")
      
    def forward(self, x):  ### [4, 128, 128, 128, 16]
        # x = F.interpolate(x, size=[ 256, 256, 32 ], mode='trilinear', align_corners=True)  
        x = F.interpolate(x, size=[ 128, 128, 32 ], mode='trilinear', align_corners=True) 
        x = self.model(x)
    
        return x  


