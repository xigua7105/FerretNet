import os
import time
import torch
import random
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from datetime import datetime
import platform
import torchvision.transforms.functional as tf


def get_cpu_info():
    system = platform.system()
    try:
        if system == "Windows":
            import subprocess
            output = subprocess.check_output(
                "wmic cpu get name", shell=True
            ).decode().strip().split("\n")
            return output[1].strip() if len(output) > 1 else "Unknown CPU"
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
            return "Unknown CPU"
        elif system == "Darwin":  # macOS
            import subprocess
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
            return output
        else:
            return "Unsupported OS"
    except Exception as e:
        return f"Error: {e}"


def get_gpu_info():
    if not torch.cuda.is_available():
        return ["No available CUDA GPU detected"]

    gpus = []
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        gpus.append(f"{name} (CUDA Capability {cap[0]}.{cap[1]})")
    return gpus


def init_cfg(cfg):
    # cudnn
    if not torch.cuda.is_available():
        print("No GPU available !")
        exit(0)

    # device info
    cfg.cpu_info = get_cpu_info()
    cfg.gpu_info = get_gpu_info()

    torch.cuda.empty_cache()
    cudnn.deterministic = True
    cudnn.benchmark = False

    # dist
    cfg.dist = True
    cfg.world_size, cfg.rank, cfg.local_rank = 1, 0, 0
    cfg.ngpus_per_node = torch.cuda.device_count()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.rank = int(os.environ["RANK"])
        cfg.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.nnodes = cfg.world_size // cfg.ngpus_per_node
    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ['SLURM_PROCID'])
        cfg.local_rank = cfg.rank % cfg.ngpus_per_node
        cfg.nnodes = cfg.world_size // cfg.ngpus_per_node
    else:
        cfg.dist = False
        cfg.nnodes = 1
    if cfg.dist:
        torch.cuda.set_device(cfg.local_rank)
        cfg.master = cfg.rank == cfg.logger.logger_rank
        cfg.dist_backend = 'nccl'
        dist.init_process_group(backend=cfg.dist_backend, init_method='env://', world_size=cfg.world_size, rank=cfg.rank)
        dist.barrier()
    else:
        cfg.master = True

    # seed
    seed = cfg.seed + cfg.local_rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # dataset
    if cfg.mode == 'train':
        cfg.trainer.batch_size_per_gpu = cfg.trainer.batch_size // cfg.world_size
        assert cfg.trainer.batch_size_per_gpu * cfg.world_size == cfg.trainer.batch_size
        cfg.trainer.batch_size_per_gpu_test = cfg.trainer.batch_size_test // cfg.world_size
        assert cfg.trainer.batch_size_per_gpu_test * cfg.world_size == cfg.trainer.batch_size_test
        cfg.trainer.num_workers = cfg.trainer.num_workers_per_gpu * cfg.world_size
    else:
        cfg.tester.batch_size_per_gpu_test = cfg.tester.batch_size_test // cfg.world_size
        assert cfg.tester.batch_size_per_gpu_test * cfg.world_size == cfg.tester.batch_size_test
        cfg.tester.num_workers = cfg.tester.num_workers_per_gpu * cfg.world_size


def distribute_bn(model, world_size, dist_bn):
    # ensure every node has the same running bn stats
    model = model.module if hasattr(model, 'module') else model
    for bn_name, bn_buf in model.named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if dist_bn == 'reduce':
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            elif dist_bn == 'broadcast':
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)
            else:
                pass


def get_timepc(cuda_synchronize=False):
    if torch.cuda.is_available() and cuda_synchronize:
        torch.cuda.synchronize()
    return time.perf_counter()


def load_state_dict(model, checkpoint_path, device='cpu'):
    print(f"Loading checkpoint: [{checkpoint_path}]")
    model_state_dict = model.state_dict()

    pretrained_dict = torch.load(checkpoint_path, map_location=device)['model']

    for k, v in pretrained_dict.items():
        if k.startswith('module.'):
            k = k[7:]
            pretrained_dict[k] = v
        else:
            pretrained_dict[k] = v

    fail_load_keys, temp_dict = [], {}
    for k, v in pretrained_dict.items():
        if k in model_state_dict.keys() and np.shape(model_state_dict[k]) == np.shape(v):
            temp_dict[k] = v
        elif k in model_state_dict.keys():
            fail_load_keys.append(k)

    model_state_dict.update(temp_dict)
    model.load_state_dict(model_state_dict)

    print('----------------------------------------------------------------------------------')
    print(
        "The number of checkpoint keys:{}\n"
        "The number of keys successfully loaded:{}\n"
        "The number of keys that failed to loaded:{}\t{}"
        .format(
            len(model_state_dict),
            len(temp_dict),
            len(fail_load_keys),
            fail_load_keys,
        )
    )
    print('----------------------------------------------------------------------------------')


def trans_state_dict(state_dict, is_dist=True):
    state_dict_modify = dict()
    if is_dist:
        for k, v in state_dict.items():
            k = k if k.startswith('module') else 'module.' + k
            state_dict_modify[k] = v
    else:
        for k, v in state_dict.items():
            k = k[7:] if k.startswith('module') else k
            state_dict_modify[k] = v
    return state_dict_modify


def save_weights(model, path, text=None, verbose=False):
    if not os.path.exists(path):
        os.makedirs(path)
    save_filename = 'epoch_%s.pth' % text if text is not None else 'model_latest.pth'
    save_path = path + '/' + save_filename
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

    if verbose:
        print('\n{:}\nsave state dict to file:{:}\n'.format(datetime.now(), save_path))


def get_net_params(model):
    num_params, frozen_num_params = 0, 0
    for p in model.parameters():
        if not p.requires_grad:
            frozen_num_params += p.numel()
        num_params += p.numel()
    return num_params, frozen_num_params

