## NuScenes
The nusc path is "/mnt/goosefs/bjcar01/algorithm/public_data/det3d/nuscenes/origin" and you can use soft link for local data.

**Download CAN bus expansion**
```
cp /mnt/cfs/algorithm/yunpeng.zhang/public_data/occupancy_infos/can_bus.zip data/
cd data && unzip can_bus.zip 
```

**Prepare nuScenes data**
Currently, we do not support the self-generation of training data for occupancy. Please use the specified datapath in the given config.

**Folder structure**
```
bevformer
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```

## Semantic KITTI
The KITTI path is "/mnt/cfs/algorithm/public_data/odometry".

**Prepare KITTI voxel label (see sh file for more details)**
```
bash process_kitti.sh
```

