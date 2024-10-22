import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml
import os
import warnings
import sys
import traceback
from contextlib import contextmanager
import logging
import psutil
import signal
import time
import gc

# suppress warning relating to deprecated pytorch functions
# Suppress the specific warning from PyTorch
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# add correct paths
if os.getcwd().endswith('fitter_3d'):  # if starting in fitter_3d dir
	os.chdir('../')
	sys.path.append('fitter_3d')

from fitter_3d.utils import load_meshes
from fitter_3d.trainer import Stage, StageManager, SMALParamGroup, SMAL3DFitter
import torch.nn as nn

parser = argparse.ArgumentParser()

parser.add_argument('--results_dir', type=str, default='fit3d_results', help="Directory in which results are stored")

# Mesh loading arguments
parser.add_argument('--mesh_dir', type=str, default='fitter_3d/ATTA_BOI',
					help="Directory (relative to SMALify) in which meshes are stored")
parser.add_argument('--frame_step', type=int, default=1,
					help="If directory is a sequence of animated frames, only take every nth frame")

# SMAL args
parser.add_argument('--shape_family_id', type=int, default=-1,
					help="Shape family to use for optimisation (-1 to use default SMAL mesh)")

# yaml src
parser.add_argument('--yaml_src', type=str, default=None, help="YAML source for experimental set-up")

# optimisation scheme to be used if .yaml not found
parser.add_argument('--scheme', type=str, default='default',
					choices=list(SMALParamGroup.param_map.keys()),
					help="Optimisation scheme")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--nits', type=int, default=100)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@contextmanager
def distributed_context(rank, world_size):
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        yield
    finally:
        dist.destroy_process_group()

def log_memory_usage(rank):
    process = psutil.Process(os.getpid())
    logging.info(f"Process {rank}: Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def timeout_handler(signum, frame):
    raise TimeoutError("Process timed out")

@contextmanager
def timeout(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def cleanup_cuda():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            obj.cpu()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def main(rank, world_size, args):
    with distributed_context(rank, world_size):
        try:
            logging.info(f"Process {rank}: Setup complete")
            log_memory_usage(rank)
            
            # try to load yaml
            yaml_loaded = False
            if args.yaml_src is not None:
                try:
                    with open(args.yaml_src) as infile:
                        yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"No YAML file found at {args.yaml_src}. Make sure this is relative to the SMALify directory.")

                yaml_loaded = True
                stage_options = yaml_cfg['stages']

                # overwrite any input args from yaml
                for arg, val in yaml_cfg['args'].items():
                    setattr(args, arg, val)

            all_mesh_names, all_target_meshes = load_meshes(mesh_dir=args.mesh_dir, frame_step=args.frame_step,
                                                                device='cpu')  # Load on CPU first

            # Ensure an even number of meshes by truncating if necessary
            total_meshes = len(all_target_meshes)
            if total_meshes % 2 != 0:
                total_meshes -= 1
                all_mesh_names = all_mesh_names[:total_meshes]
                all_target_meshes = all_target_meshes[:total_meshes]

            # Calculate the number of meshes per GPU
            meshes_per_gpu = total_meshes // world_size
            start_idx = rank * meshes_per_gpu
            end_idx = start_idx + meshes_per_gpu

            # Distribute meshes to each GPU
            mesh_names = all_mesh_names[start_idx:end_idx]
            device = torch.device(f'cuda:{rank}')  # Create a proper torch.device object
            target_meshes = all_target_meshes[start_idx:end_idx].to(device)

            n_batch = meshes_per_gpu
            
            logging.info(f"GPU {rank}: Processing {len(mesh_names)} target meshes (indices {start_idx} to {end_idx-1})")
            
            # Synchronize all processes to ensure all print statements are executed before continuing
            dist.barrier()

            os.makedirs(args.results_dir, exist_ok=True)
            manager = StageManager(out_dir=args.results_dir, labels=mesh_names)

            smal_model = SMAL3DFitter(batch_size=n_batch,
                                      device=device, shape_family=args.shape_family_id)

            # Move model to the correct device
            smal_model = smal_model.to(device)
            
            # Wrap the model with DistributedDataParallel
            smal_model = DDP(smal_model, device_ids=[rank])

            stage_kwargs = dict(target_meshes=target_meshes, smal_3d_fitter=smal_model,
                                out_dir=args.results_dir, device=device,
                                mesh_names=mesh_names)

            # if provided, load stages from YAML
            if yaml_loaded:
                for stage_name, kwargs in stage_options.items():
                    stage = Stage(name=stage_name, **kwargs, **stage_kwargs)
                    manager.add_stage(stage)

            # otherwise, load from arguments
            else:
                logging.info("No YAML provided. Loading from system args. ")
                stage = Stage(scheme=args.scheme, nits=args.nits, lr=args.lr,
                              **stage_kwargs)
                manager.add_stage(stage)

            logging.info(f"Process {rank}: Model and data loaded")
            log_memory_usage(rank)

            manager.run()
            manager.plot_losses('losses')  # plot to results file

            logging.info(f"Process {rank}: Completed successfully")
            log_memory_usage(rank)
        except Exception as e:
            logging.error(f"Process {rank}: Error occurred")
            logging.error(f"Error message: {str(e)}")
            logging.error("Traceback:")
            traceback.print_exc()
        finally:
            # Explicit cleanup
            cleanup_cuda()
            logging.info(f"Process {rank}: Cleanup complete")
            log_memory_usage(rank)

def run_processes(world_size, args):
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=main, args=(rank, world_size, args))
        p.start()
        processes.append(p)

    for p in processes:
        try:
            with timeout(300):  # 5 minutes timeout
                p.join()
        except TimeoutError:
            logging.error(f"Process {p.pid} timed out, terminating...")
            p.terminate()
            p.join()

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Uncomment the following line to use only one GPU
    # world_size = 1
    world_size = torch.cuda.device_count()
    
    try:
        run_processes(world_size, args)
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        traceback.print_exc()
    finally:
        # Ensure all CUDA operations are completed
        cleanup_cuda()
        logging.info("Main process: Final cleanup complete")
