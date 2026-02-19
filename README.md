# SMILify

This repository is based on [SMALify](https://github.com/benjiebob/SMALify) with the aim to turn any rigged 3D model into
a SMAL compatible model. There are Blender files to convert your mesh and lots of code
changes to deal with arbitrary armature configurations, rather than assuming a fixed
quadruped model.

For now, I'll focus on insects, hence **SMIL**.

## Neural Inference Examples

Multi-view 3D reconstruction using neural inference:

<img src="docs/mouse_18_cam_smil_multi.gif" width="800"> <img src="docs/mouse_18_cam_smil.gif" width="800">

Example 18 camera inference results, using a newly developed [parametric mouse model](3D_model_prep/SMILy_Mouse_static_joints_Falkner_conv_repose_hind_legs.pkl)

<img src="docs/peruphasma_4_cam_smil.gif" width="800">

Example 4-5 camera inference results with a molde trained on data collected from an Omni-Directional Treadmill (ODT) using a [parametric multi species stick insect model](3D_model_prep/SMILy_STICK.pkl) configured with the [Blender SMIL Addon](3D_model_prep/SMIL_processing_addon.py).


## Installation
1. Clone the repository **with submodules** and enter directory
   ```
   git clone https://github.com/FabianPlum/SMILify
   ```
   Note: If you don't clone with submodules you won't get the sample data from BADJA/StanfordExtra/SMALST.

2. install pytorch (and Co.)
   ```
   conda create -n pytorch3d python=3.10
   conda activate pytorch3d
   conda install pytorch=2.3.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install -c conda-forge -c fvcore iopath ninja imageio scikit-image
   pip install yacs pycocotools
   pip install --upgrade iopath
   ```

3. clone pytorch3d and install (WINDOWS)
   
   **NOTE**: I've never gotten this to work properly on Windows. I'd recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) or making an Ubuntu partition.
   If by what ever dark magic you posess you manage to run this on Win11, please open a PR and share your arcane wisdom.
   ```
   git clone https://github.com/facebookresearch/pytorch3d.git
   cd pytorch3d
   pip install -e .
   cd ..
   ```
   
   on LINUX just run
   ```
   conda install pytorch3d -c pytorch3d
   ```

4. some more dependencies
   ```
   pip install matplotlib scipy opencv-python nibabel trimesh timm pytest h5py psutil
   ```
   
5. Test your installation
   ```
   pytest tests/ -v -s
   ```

## Dataset preprocessing
_coming soon_

## Training
_coming soon_

## Inference
_coming soon_

## Benchmarking
_coming soon_


# Code refactor TODOs 
- [X] Move all legcay funcitonality and documentation to it's own sub-directory to clean up the repo and make its purpose more apparent.
- [ ] Remove all currently used recursive clones. The repo should work on its own without the need of cloning submodules.
- [ ] If a submodule is needed, we should re-write it and add it to an appropriate subfolder. Otherwise, this repo is entirely un-maintainable.
- [ ] At the moment, the SMAL models require 2 to 3 separate types of data files as well as hard-coded priors for the joint limits. These should be handled more gracefully, like in the new SMIL implementation. All model info should be contained in a single, readable and editable file.
- [X] Get rid of the numpy/chumpy dependency mess.
- [X] Allow importing legacy SMAL models with chumpy variables WITHOUT requiring chumpy to be installed through custom unpickler.
- [ ] Write a conversion script from the old SMAL format consisting of multiple files into our new single file structure containing all the data. I don't care if the files are large, as long as they are readable and first and foremost editable.
- [ ] The code is poorly documented. That needs to be fixed.
- [X] The code is poorly tested. That needs to be fixed. Write integration tests for main functionality.
- [ ] Let's see how far we can get with this in our limited time BUT I would love to re-write this whole thing as a Blender addon. But that's for another day (probably more of a "project wish" than related to refactoring).

## Functionality / broader project TODOs
- [ ] Allow to add user-defined priors for joint limits in the Blender addon.
- [X] Finish cleaning antscan dataset and prepare models for fitting.
- [ ] Create SMIL model from massive antscan dataset.
- [X] Add configurable mouse SMIL model.
- [X] Re-implement multi-GPU mesh registration cleanly.

## Acknowledgements
- [SMALify](https://github.com/benjiebob/SMALify); Biggs et al, the original repo on which this one is based.
This repository owes a great deal to the following works and authors:
- [SMAL](http://smal.is.tue.mpg.de/); Zuffi et al. designed the SMAL deformable quadruped template model and have been wonderful for providing advice throughout my animal reconstruction PhD journey.
- [SMPLify](http://smplify.is.tue.mpg.de/); Bogo et al. provided the basis for our original ChumPY implementation and inspired the name of this repo.
- [SMALST](https://github.com/silviazuffi/smalst); Zuffi et al. provided a PyTorch implementations of the SMAL skinning functions which have been used here.


## Contribute
Please create a pull request or submit an issue if you would like to contribute.

## Licensing
(c) Fabian Plum, Imperial College London & Forschungs Zentrum Juelich & scAnt UG

By downloading this codebase and included dataset(s), you agree to the [Creative Commons Attribution-NonCommercial 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/). This license allows users to use, share and adapt the codebase and dataset(s), so long as credit is given to the authors (e.g. by citation) and the dataset is not used for any commercial purposes.

THIS SOFTWARE AND ANNOTATIONS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

