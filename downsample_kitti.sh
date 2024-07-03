cd $(readlink -f `dirname $0`)
source /mnt/cfs/algorithm/yunpeng.zhang/.bashrc
conda activate bevformer

export PYTHONPATH="."
export OMP_NUM_THREADS=8

kitti_root=/mnt/cfs/algorithm/public_data/odometry/kitti_odometry_color
kitti_preprocess_root=/mnt/cfs/algorithm/public_data/odometry/semkitti_multiscale_voxel_labels
data_info_path=./tools/data_converter/kitti_process/semantic-kitti.yaml

python ./tools/data_converter/kitti_process/semantic_kitti_downsample.py \
--kitti_root $kitti_root \
--kitti_preprocess_root $kitti_preprocess_root \
--data_info_path $data_info_path
