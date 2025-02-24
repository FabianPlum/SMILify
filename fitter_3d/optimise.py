import torch
import argparse
import yaml
import os
import warnings
import sys
import numpy as np
from math import ceil
import gc
import glob

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
import config

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

def clear_cuda_memory():
	"""Clear CUDA memory and garbage collect."""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	gc.collect()

def combine_stage_results(results_dir, stage_names, n_batches):
	"""Combines stage results from multiple batches into single files."""
	for stage_name in stage_names:
		combined_data = None
		
		# Load and combine data from each batch
		for batch_idx in range(n_batches):
			batch_filename = f"{stage_name}_batch_{batch_idx}.npz"
			batch_path = os.path.join(results_dir, batch_filename)
			
			if not os.path.exists(batch_path):
				continue
				
			batch_data = np.load(batch_path)
			batch_dict = dict(batch_data)
			
			if combined_data is None:
				combined_data = batch_dict
			else:
				# Concatenate arrays for each parameter
				for key in combined_data:
					if key != 'faces':  # faces should be the same for all batches
						combined_data[key] = np.concatenate([combined_data[key], batch_dict[key]], axis=0)
		
		if combined_data is not None:
			# Save combined results
			output_path = os.path.join(results_dir, f"{stage_name}.npz")
			np.savez(output_path, **combined_data)
			
			# Clean up batch files
			for batch_idx in range(n_batches):
				batch_path = os.path.join(results_dir, f"{stage_name}_batch_{batch_idx}.npz")
				if os.path.exists(batch_path):
					os.remove(batch_path)

def get_mesh_files(mesh_dir, frame_step=1):
	"""Get list of mesh files from directory."""
	# Get all .obj files in the directory
	mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))
	# Apply frame step if specified
	mesh_files = mesh_files[::frame_step]
	return mesh_files

def main(args):
	# try to load yaml
	yaml_loaded = False
	if args.yaml_src is not None:
		try:
			with open(args.yaml_src) as infile:
				yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
		except FileNotFoundError:
			raise FileNotFoundError(
				"No YAML file found at {args.yaml_src}. Make sure this is relative to the SMALify directory.")

		yaml_loaded = True
		stage_options = yaml_cfg['stages']

		# overwrite any input args from yaml
		for arg, val in yaml_cfg['args'].items():
			setattr(args, arg, val)

	# Get list of mesh files
	mesh_files = get_mesh_files(args.mesh_dir, args.frame_step)
	n_total = len(mesh_files)
	batch_size = config.SPLIT_TARGET_MESHES_INTO_BATCHES_OF_SIZE
	
	if batch_size <= 0:
		batch_size = n_total
	
	n_batches = ceil(n_total / batch_size)
	stage_names = []  # Keep track of stage names for combining results later
	
	try:
		for batch_idx in range(n_batches):
			print(f"Processing batch {batch_idx + 1} of {n_batches}")
			start_idx = batch_idx * batch_size
			end_idx = min((batch_idx + 1) * batch_size, n_total)
			
			# Get batch file names and load meshes for this batch only
			batch_files = mesh_files[start_idx:end_idx]
			batch_mesh_names = [os.path.basename(f) for f in batch_files]
			
			# Load meshes for just this batch
			_, batch_target_meshes = load_meshes(mesh_files=batch_files, device=device)
			
			n_batch = len(batch_target_meshes)
			os.makedirs(args.results_dir, exist_ok=True)
			manager = StageManager(out_dir=args.results_dir, labels=batch_mesh_names)

			smal_model = SMAL3DFitter(batch_size=n_batch,
									  device=device, shape_family=args.shape_family_id)

			stage_kwargs = dict(target_meshes=batch_target_meshes, smal_3d_fitter=smal_model,
								out_dir=args.results_dir, device=device,
								mesh_names=batch_mesh_names)

			# if provided, load stages from YAML
			if yaml_loaded:
				for stage_name, kwargs in stage_options.items():
					batch_stage_name = f"{stage_name}_batch_{batch_idx}" if n_batches > 1 else stage_name
					stage = Stage(name=batch_stage_name, **kwargs, **stage_kwargs)
					manager.add_stage(stage)
					if batch_idx == 0:  # Only add stage names once
						stage_names.append(stage_name)

			# otherwise, load from arguments
			else:
				batch_stage_name = f"stage_batch_{batch_idx}" if n_batches > 1 else "stage"
				stage = Stage(scheme=args.scheme, nits=args.nits, lr=args.lr,
							  name=batch_stage_name, **stage_kwargs)
				manager.add_stage(stage)
				if batch_idx == 0:
					stage_names.append("stage")

			manager.run()
			# Only create losses.png when processing all meshes simultaneously
			if n_batches > 1:
				manager.plot_losses(f'losses_batch_{batch_idx}')
			else:
				manager.plot_losses('losses')
			
			# Clean up CUDA memory after each batch
			del smal_model
			del manager
			del batch_target_meshes
			clear_cuda_memory()
		
		# If we split into batches, combine the results
		if n_batches > 1:
			combine_stage_results(args.results_dir, stage_names, n_batches)
		
	finally:
		# Ensure CUDA memory is cleared even if an error occurs
		clear_cuda_memory()

if __name__ == "__main__":
	args = parser.parse_args()
	main(args)
