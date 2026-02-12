"""
Test configuration overrides.

Imports the main config and overrides only what's needed for fast testing:
- Uses SMPL_fit.pkl (OmniAnt) model explicitly
- Reduces optimizer iterations to 10 per stage
"""
from config import *
from os.path import join

# Use OmniAnt model for testing, regardless of what config.py currently uses
SMAL_FILE = join("3D_model_prep", 'SMPL_fit.pkl')

# Reduce iterations for fast testing (10 per stage instead of full run)
OPT_WEIGHTS[7] = [10, 10, 10, 10]
