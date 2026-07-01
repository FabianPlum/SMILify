module purge
module load Stages/2024 # -> needed to ensure we use Python <= 3.11
module load GCC OpenMPI
# Some base modules commonly used in AI
module load IPython git
#module load Flask Seaborn

# ML Frameworks
#module load  PyTorch scikit-learn torchvision PyTorch-Lightning
#module load tensorboard
#module load h5py