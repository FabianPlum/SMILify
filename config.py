from os.path import join
import pickle as pkl
import cv2
import time
import numpy as np

# Define paths to each dataset
REPLICANT_PATH = "data/replicAnt_trials/SMIL_COCO"
OUTPUT_DIR = "checkpoints/{0}".format(time.strftime("%Y%m%d-%H%M%S"))

CROP_SIZE = 512  # image resolution for output
VIS_FREQUENCY = 50  # every how many iterations the model plots are to be generated
GPU_IDS = "0"  # GPU number to run on (not applicable for CPU)

# Run settings (I wouldn't recommend changing these unless you have good reason)
ALLOW_LIMB_SCALING = True  # Allow scaling parameters, see Who Left the Dogs Out?
SPLIT_TARGET_MESHES_INTO_BATCHES_OF_SIZE = (
    100  # Split target meshes into batches of size N. -1 means process all simultaneously
)

# Sequence/Image Settings
SHAPE_FAMILY = -1  # -1 for SMIL models (legacy SMAL shape families are no longer supported)

SEQUENCE_OR_IMAGE_NAME = "replicAnt:SMIL_09_synth.jpg"
WINDOW_SIZE = 10  # Changed number of frames processed in one go.

# custom elements added:
# SMAL_FILE = join("3D_model_prep", 'smpl_ATTA.pkl') #  BASE ANT MODEL
SMAL_FILE = join("3D_model_prep", "SMIL_OmniAnt.pkl")  # LATEST TEXTURED ANT MODEL WITH ALL VARIATION
# SMAL_FILE = join("3D_model_prep", 'SMILy_STICK.pkl') # LATEST STICK INSECT MODEL
# SMAL_FILE = join("3D_model_prep", 'SMILy_Mouse_static_joints_Falkner_conv_repose_hind_legs_fix_eyes.pkl') # LATEST MOUSE MODEL

ignore_sym = True  # ignore provided symmetry file, when using custom models
PLOT_RESULTS = True  # only applies to 3D fitting (fitter_3d/optimise.py)
DEBUG = False  # use to get A LOT of "useful" messages

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
    [9e-2, 5e-3, 5e-4, 2e-4],
]  # Learning Rate

# ignore joints (in case annotation conventions differ)
IGNORE_JOINTS = ["b_t", "b_a_4", "b_a_5"]  # temporary
# -> replicAnt data places these in different locations then SMIL, see also smal_fitter/priors/joint_limits.py


def set_model(smal_file):
    """Load the model .pkl and (re-)derive every model-dependent global from it.

    The model hierarchy (joint names, kinematic tree, root joint, pose/shape sizes) is read
    from the .pkl rather than hard-coded. Called at import with the default SMAL_FILE; use
    smal_fitter.neuralSMIL.configs.config_utils.apply_smal_file_override to switch models.
    """
    global SMAL_FILE, dd, STATIC_JOINT_LOCATIONS, joint_names, ROOT_JOINT, TORSO_JOINTS
    global CANONICAL_MODEL_JOINTS, MARKER_TYPE, MARKER_COLORS, N_POSE, N_BETAS

    # read in smpl file
    with open(smal_file, "rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    SMAL_FILE = smal_file

    # check if joint locations are static
    if "static_joint_locs" in dd and dd["static_joint_locs"]:
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
    TORSO_JOINTS = [
        i for i, elem in enumerate(joint_names) if elem in ["b_a_1", "l_1_co_r", "l_1_co_l", "b_h", "ma_l", "ma_r"]
    ]

    # all joints
    CANONICAL_MODEL_JOINTS = [i for i in range(len(joint_names))]

    # same for all joints
    MARKER_TYPE = [cv2.MARKER_STAR for i in range(len(CANONICAL_MODEL_JOINTS))]

    # make it a fun rainbow, no specifics
    MARKER_COLORS = [
        [int(255 - i * 255 / len(CANONICAL_MODEL_JOINTS)), int(i * 255 / len(CANONICAL_MODEL_JOINTS)), 100]
        for i in range(len(CANONICAL_MODEL_JOINTS))
    ]

    N_POSE = len(joint_names) - 1  # not including global rotation
    try:
        N_BETAS = dd["shapedirs"].shape[2]  # number of SMAL shape parameters to optimize over
        if DEBUG:
            print("INFO: Found custom betas:", N_BETAS)
    except (IndexError, KeyError):
        # if no learned betas / shapedirs are provided default to 20
        N_BETAS = 20
        if DEBUG:
            print("INFO: Using default num betas of", N_BETAS)


set_model(SMAL_FILE)
