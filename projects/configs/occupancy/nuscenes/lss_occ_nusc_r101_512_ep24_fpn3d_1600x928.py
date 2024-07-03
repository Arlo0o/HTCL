_base_ = [
    '../../datasets/custom_nus-3d.py',
    '../../_base_/default_runtime.py'
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [256, 256, 16]
# downsample ratio in [x, y, z] when generating 3D volumes in LSS
lss_downsample = [1, 1, 1]
voxel_channels = [80, 160, 320, 640]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (928, 1600),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}
numC_Trans = 80
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

model = dict(
    type='BEVDepthOccupancy',
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(type='ViewTransformerLiftSplatShootVoxel',
                              loss_depth_weight=3.,
                              loss_depth_type='kld',
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=numC_Trans,
                              vp_megvii=False),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
    ),
    img_bev_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=17,
        loss_voxel_prototype='cylinder3d',
        supervise_points=True,
        sampling_img_feats=True,
        soft_weights=True,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1,
            loss_voxel_lovasz_weight=1,
            loss_point_ce_weight=float(len(voxel_out_indices)),
            loss_point_lovasz_weight=float(len(voxel_out_indices)),
        ),
    ),
    train_cfg=dict(pts=None),
    test_cfg=dict(pts=None),
)

dataset_type = 'CustomNuScenesOccLSSDataset'
data_root = '/mnt/cfs2/algorithm/public_data/det3d/nuscenes/origin/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config,
                sequential=False, aligned=True, trans_only=False, data_root=data_root,
                mmlabnorm=True, load_depth=True, img_norm_cfg=img_norm_cfg),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        speed_mode='abs_speed'),
    # how to handle bda
    dict(type='LoadSemanticPoint'),
    dict(type='CreateVoxelLabels', point_cloud_range=point_cloud_range, grid_size=occ_size, unoccupied=0),
    dict(type='MultiViewProjections'),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points_occ', 'points_uv'],
            meta_keys=['pc_range', 'occ_size']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config,
         sequential=False, aligned=True, trans_only=False, mmlabnorm=True, 
         data_root=data_root, img_norm_cfg=img_norm_cfg),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        speed_mode='abs_speed',
        is_train=False),
    dict(type='LoadSemanticPoint'),
    dict(type='CreateVoxelLabels', point_cloud_range=point_cloud_range, grid_size=occ_size, unoccupied=0),
    dict(type='MultiViewProjections'),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points_occ', 'points_uv'],
            meta_keys=['pc_range', 'occ_size']),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='/mnt/cfs/algorithm/yunpeng.zhang/public_data/occupancy_infos/nuscenes_infos_temporal_val.pkl',
    bevdepth_ann_file='/mnt/cfs/algorithm/junjie.huang/data/bevdet-nuscenes_infos_val.pkl',
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    queue_length=1,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        bevdepth_ann_file='/mnt/cfs/algorithm/junjie.huang/data/bevdet-nuscenes_infos_train.pkl',
        ann_file='/mnt/cfs/algorithm/yunpeng.zhang/public_data/occupancy_infos/nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        queue_length=1,
        use_semantic=True,
        box_type_3d='LiDAR'),
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='lidarseg_mean',
    rule='greater',
)