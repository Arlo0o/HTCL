import os
import json

# from easydict import EasyDict
# from submit import get_config
import socket
import argparse

def get_work_index():
    while True:
        try:
            addr = os.environ.get("MASTER_ADDR", "{}")  # .replace('master','worker')
            if addr == "localhost":  # 当为master节点时，需要通过hostname获取IP，不然获取IP为127.0.0.1
                addr = os.environ.get("HOSTNAME", "{}")
            if "phigent.io" in addr:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("114.114.114.114", 80))
                master_addr = s.getsockname()[0]
            else:
                master_addr = socket.gethostbyname(addr)  # 获取master IP地址
            master_port = os.environ.get("MASTER_PORT", "{}")  # 获取master port
            print("MASTER_ADDR: %s", addr)
            world_size = os.environ.get("WORLD_SIZE", "{}")  # job 的总进程数
            rank = os.environ.get("RANK", "{}")  #  当前进程的进程号, 必须在 rank==0 的进程内保存参数
            # logging.info("RANK: %s", rank)
            break
        except:
            print("get 'TF_CONFIG' failed, sleep for 1 second~")
            os.system("sleep 1s")
            continue

    return int(world_size), int(rank), master_addr, int(master_port)


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("--config", default=None, help="test config file path")
    parser.add_argument("--gpu", default=8, type=int)
    parser.add_argument("--resume-from", default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = args.config
    assert config is not None

    os.system("sleep 10s")
    cfg = dict(
        cluster=dict(
            cluster_name="tc",
            num_gpus=args.gpu,
        ),
        config_file=config,
        no_validate=True,
        resume_from=args.resume_from,
    )

    num_nodes, node_rank, master_addr, master_port = get_work_index()
    pwd_dict = {
        "tc": "/mnt/cfs/algorithm/yunpeng.zhang/codes/BEVFormer",
    }
    PWD = pwd_dict[cfg["cluster"]["cluster_name"]]

    # pwd + config, num_gpu, master_addr, num_nodes, node_rank
    CMD = "bash tools/dist_train_mpi_new.sh %s %d %s %d %d %d " % (
        PWD + "/" + cfg["config_file"],
        cfg["cluster"]["num_gpus"],
        master_addr,
        num_nodes,
        node_rank,
        master_port,
    )
    cfg_keys = ["resume_from", "work_dir", "no_validate", "load_from"]
    for key in cfg_keys:
        key_val = cfg.get(key, None)
        if key_val is None:
            continue
        if key in ["no_validate"]:
            CMD += "--" + key.replace("_", "-") + " "
        else:
            CMD += "--" + key.replace("_", "-") + " " + str(cfg[key]) + " "

    # start train
    print(CMD)
    os.system("cd %s && %s" % (PWD, CMD))
