cd $(readlink -f `dirname $0`)
# source /mnt/cfs/algorithm/yunpeng.zhang/.bashrc
# conda activate bevformer

echo $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

export PYTHONPATH="."
export OMP_NUM_THREADS=8

ckpt=$2
python tools/test.py $config $ckpt --eval mAP
