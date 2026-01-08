from os.path import join
import pickle as pkl
import cv2
import os
import time
import numpy as np

# Define paths to each dataset
data_path = "data"
BADJA_PATH = "data/BADJA"
STANFORD_EXTRA_PATH = "data/StanfordExtra"
REPLICANT_PATH = "data/replicAnt_trials/SMIL_COCO"
OUTPUT_DIR = "checkpoints/{0}".format(time.strftime("%Y%m%d-%H%M%S"))

CROP_SIZE = 512  # image resolution for output
VIS_FREQUENCY = 50  # every how many iterations the model plots are to be generated
GPU_IDS = "0"  # GPU number to run on (not applicable for CPU)

# Run settings (I wouldn't recommend changing these unless you have good reason)
FORCE_SMAL_PRIOR = False  # Allow the more recent Unity-based prior for dogs.
ALLOW_LIMB_SCALING = True  # Allow scaling parameters, see Who Left the Dogs Out?
SPLIT_TARGET_MESHES_INTO_BATCHES_OF_SIZE = 100  # Split target meshes into batches of size N. -1 means process all simultaneously
# moved up the following line to have in one place with other prior data
UNITY_SHAPE_PRIOR = join(data_path, 'priors', 'unity_betas.npz')

# Sequence/Image Settings
SHAPE_FAMILY = -1  # Choose from Cat (e.g. House Cat/Tiger/Lion), Canine (e.g. Dog/Wolf), Equine (e.g. Horse/Zebra), Bovine (e.g. Cow), Hippo
# SEQUENCE_OR_IMAGE_NAME = "badja:rs_dog"
# SEQUENCE_OR_IMAGE_NAME = "stanfordextra:n02099601-golden_retriever/n02099601_176.jpg"

SEQUENCE_OR_IMAGE_NAME = "replicAnt:SMIL_09_synth.jpg"
IMAGE_RANGE = range(0, 1)  # Frames to process from sequence. Ignored for stanford extra
WINDOW_SIZE = 10  # Changed number of frames processed in one go.

# Generate video settings
EPOCH_NAME = "st10_ep0"  # convention used for the final output file. Don't change this without good reason.

# SMAL
SMAL_MODEL_PATH = join(data_path, 'SMALST', 'smpl_models')
#SMAL_FILE = join(SMAL_MODEL_PATH, 'my_smpl_00781_4_all.pkl')

# custom elements added:
# SMAL_FILE = join("3D_model_prep", 'smpl_ATTA.pkl')
#SMAL_FILE = join("3D_model_prep", 'SMIL_OmniAnt.pkl') # USING LATEST SMIL DATA
#SMAL_FILE = join("3D_model_prep", 'SMILy_STICK.pkl')
SMAL_FILE = join("3D_model_prep", 'SMILy_Mouse_static_joints_REPOSE_LONG.pkl') #UPDATED JOINT LOCS

ignore_sym = True  # ignore provided symmetry file, when using custom models
ignore_hardcoded_body = True  # ignore model joints in config file and use what's contained in the SMPL file
PLOT_RESULTS = True  # only applies to 3D fitting (fitter_3d/optimise.py)
DEBUG = False  # use to get A LOT of "useful" messages

# DATALOADER
IMG_RES = 512

# RENDERER
MESH_COLOR = [0, 172, 223]

# OPTIMIZER - You may need to adjust these depending on the sequence.
OPT_WEIGHTS = [
    [25.0, 10.0, 7.5, 5.0],  # Joint
    [0.0, 500.0, 1000.0, 1000.0],  # Sil Reproj
    [0.0, 1.0, 1.0, 1.0],  # Betas
    [0.0, 1.0, 1.0, 1.0],  # Pose
    [0.0, 100.0, 100.0, 100.0],  # Joint limits, suppressed in the original SMALify code!
    [0.0, 0.1, 0.1, 0.1],  # Splay
    [500.0, 100.0, 100.0, 100.0],  # Temporal
    [600, 400, 600, 600],  # Num iterations
    [9e-2, 5e-3, 5e-4, 2e-4]]  # Learning Rate

if ignore_hardcoded_body:
    # this is NOT a great place for reading this in, but unless I hide the hard-coded dog stuff elsewhere,
    # this is the only low-effort way to get the info dynamically without making bigger changes in smal_torch.py and co

    # read in smpl file
    with open(SMAL_FILE, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        dd = u.load()

    # check if joint locations are static
    if "static_joint_locs" in dd and dd["static_joint_locs"] == True:
        STATIC_JOINT_LOCATIONS = True
        if DEBUG:
            print("INFO: Joint locations are static")
    else:
        STATIC_JOINT_LOCATIONS = False
        if DEBUG:
            print("INFO: Joint locations are not static")

    # get J_names  |  names of joints
    joint_names = dd["J_names"]
    if DEBUG:
        print("INFO: Joint names:", joint_names)

    ROOT_JOINT = dd["J_names"][np.where(dd["kintree_table"][0] == -1)[0][0]]
    if DEBUG:
        print("INFO: Root joint:", ROOT_JOINT)

    # IDs of every joint that starts with b, referring to the animal body, including the tail
    # as this is used for the initial alignment, we include the mandibles as well to provide a sense of left vs right
    TORSO_JOINTS = [i for i, elem in enumerate(joint_names) if elem in ["b_a_1", "l_1_co_r", "l_1_co_l",
                                                                        "b_h", "ma_l", "ma_r"]]

    # exclude wings
    WING_JOINTS = [i for i, elem in enumerate(joint_names) if elem.split("_")[0] == "w"]

    # all joints
    CANONICAL_MODEL_JOINTS = [i for i in range(len(joint_names))]

    # ignore joints (in case annotation conventions differ)
    IGNORE_JOINTS = ["b_t", "b_a_4", "b_a_5"]  # temporary
    # -> replicAnt data places these in different locations then SMIL, see also smal_fitter/priors/joint_limits.py

    # same for all joints
    MARKER_TYPE = [cv2.MARKER_STAR for i in range(len(CANONICAL_MODEL_JOINTS))]

    # make it a fun rainbow, no specifics
    MARKER_COLORS = [[int(255 - i * 255 / len(CANONICAL_MODEL_JOINTS)),
                      int(i * 255 / len(CANONICAL_MODEL_JOINTS)),
                      100]
                     for i in
                     range(len(CANONICAL_MODEL_JOINTS))]

    N_POSE = len(joint_names) - 1  # not including global rotation
    # WARNING -> Now overwritten in trainer.py line 78
    try:
        N_BETAS = dd["shapedirs"].shape[2]  # number of SMAL shape parameters to optimize over
        if DEBUG:
            print("INFO: Found custom betas:", N_BETAS)
    except IndexError or KeyError:
        # if no learned betas / shapedirs are provided default to 20
        N_BETAS = 20
        if DEBUG:
            print("INFO: Using default num betas of", N_BETAS)


else:  # use joint and plotting configuration of default dog model:

    if os.name == 'nt':
        ## If WINDOWS
        SMAL_DATA_FILE = join(SMAL_MODEL_PATH, 'my_smpl_data_00781_4_all_WIN.pkl')
        SMAL_UV_FILE = join(SMAL_MODEL_PATH, 'my_smpl_00781_4_all_template_w_tex_uv_001_WIN.pkl')
        SMAL_SYM_FILE = join(SMAL_MODEL_PATH, 'symIdx_WIN.pkl')

        if not os.path.exists(SMAL_FILE):
            print("Unable to find Windows file. Please run ./utilities/convert_smal_windows.ps1")

        # PRIORS
        WALKING_PRIOR_FILE = join(data_path, 'priors', 'walking_toy_symmetric_pose_prior_with_cov_35parts_WIN.pkl')
    else:
        SMAL_DATA_FILE = join(SMAL_MODEL_PATH, 'my_smpl_data_00781_4_all.pkl')
        SMAL_UV_FILE = join(SMAL_MODEL_PATH, 'my_smpl_00781_4_all_template_w_tex_uv_001.pkl')
        SMAL_SYM_FILE = join(SMAL_MODEL_PATH, 'symIdx.pkl')

        # PRIORS
        WALKING_PRIOR_FILE = join(data_path, 'priors', 'walking_toy_symmetric_pose_prior_with_cov_35parts.pkl')

    # JOINT DEFINITIONS
    TORSO_JOINTS = [2, 5, 8, 11, 12, 23]

    CANONICAL_MODEL_JOINTS = [
        10, 9, 8,  # upper_left [paw, middle, top]
        20, 19, 18,  # lower_left [paw, middle, top]
        14, 13, 12,  # upper_right [paw, middle, top]
        24, 23, 22,  # lower_right [paw, middle, top]
        25, 31,  # tail [start, end]
        33, 34,  # ear base [left, right]
        35, 36,  # nose, chin
        38, 37,  # ear tip [left, right]
        39, 40,  # eyes [left, right]
        15, 15,  # withers, throat (TODO: Labelled same as throat for now), throat
        28]  # tail middle

    # indicate invalid joints (i.e. not labelled) by -1
    BADJA_ANNOTATED_CLASSES = [
        14, 13, 12,  # upper_left [paw, middle, top]
        24, 23, 22,  # lower_left [paw, middle, top]
        10, 9, 8,  # upper_right [paw, middle, top]
        20, 19, 18,  # lower_right [paw, middle, top]
        25, 31,  # tail [start, end] (note, missing the tail middle point)
        -1, -1,  # ear base [left, right]
        33, -1,  # nose, chin (note, missing the 'jaw base' point)
        36, 35,  # ear tip [left, right]
        -1, -1,  # eyes [left, right]
        -1, 15,  # withers, throat
        28]  # tail middle

    # Visualization
    MARKER_TYPE = [
        cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_STAR, cv2.MARKER_CROSS,  # upper_left
        cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_STAR, cv2.MARKER_CROSS,  # lower_left
        cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_STAR, cv2.MARKER_CROSS,  # upper_right
        cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_STAR, cv2.MARKER_CROSS,  # lower_right
        cv2.MARKER_CROSS, cv2.MARKER_TRIANGLE_DOWN,  # tail
        cv2.MARKER_CROSS, cv2.MARKER_CROSS,  # right_ear, left_ear
        cv2.MARKER_CROSS, cv2.MARKER_STAR,  # nose, chin
        cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_TRIANGLE_DOWN,  # right_tip, left_tip
        cv2.MARKER_CROSS, cv2.MARKER_CROSS,  # right_eye, left_eye
        cv2.MARKER_CROSS, cv2.MARKER_CROSS,  # withers, throat
        cv2.MARKER_STAR]  # tail middle

    MARKER_COLORS = [
        [230, 25, 75], [230, 25, 75], [230, 25, 75],  # upper_left, red
        [255, 255, 25], [255, 255, 25], [255, 255, 25],  # lower_left, yellow
        [60, 180, 75], [60, 180, 75], [60, 180, 75],  # upper_right, green
        [0, 130, 200], [0, 130, 200], [0, 130, 200],  # lower_right, blue
        [240, 50, 230], [240, 50, 230],  # tail, majenta
        [255, 153, 204], [29, 98, 115],  # left_ear, pink & right_ear, turquoise
        [245, 130, 48], [245, 130, 48],  # nose, chin
        [255, 153, 204], [29, 98, 115],  # left_ear, pink & right_tip, turquoise
        [0, 0, 0], [0, 0, 0],  # right eye, left eye: black
        [128, 0, 0], [128, 0, 0],  # withers, throat, maroon
        [240, 50, 230]]  # tail middle

    # TODO - remove N_POSE variable, as the number of joints should be taken from the input SMPL file instead of hard-coded
    N_POSE = 34  # not including global rotation
    # WARNING -> Now overwritten in trainer.py line 78
    N_BETAS = 20  # number of SMAL shape parameters to optimize over

    joint_names = [f"joint_{i}" for i in range(N_POSE)]
