cd $(readlink -f `dirname $0`)
source /mnt/cfs/algorithm/yunpeng.zhang/.bashrc
conda activate bevformer

echo $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

export PYTHONPATH="."

ckpt=$2
bash tools/dist_test.sh $config $ckpt $3