"""Legacy chumpy-aware pickle loader for SMAL/SMIL model files."""

import pickle

import numpy as np

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles legacy SMAL model files containing chumpy arrays"""

    def __init__(self, file, encoding="latin1"):
        """Initialize with latin1 encoding to handle legacy pickle files"""
        super().__init__(file, encoding=encoding)

    def find_class(self, module, name):
        """Override class lookup to handle chumpy arrays"""
        if module == "chumpy.ch" and name == "Ch":
            return self.ChumpyWrapper
        return super().find_class(module, name)

    class ChumpyWrapper:
        """Wrapper class that mimics chumpy array behavior but stores only numpy arrays"""

        def __init__(self, *args, **kwargs):
            """Initialize with data from args or empty array"""
            self.data = np.array(args[0]) if args else np.array([])

        def __array__(self):
            """Allow numpy array conversion via np.array(instance)"""
            return self.data

        def __setstate__(self, state):
            """Handle unpickling of chumpy arrays in various formats"""
            if isinstance(state, dict):
                # Handle old chumpy format where data is stored in 'x' key
                self.data = np.array(state.get("x", []))
            else:
                # Handle both tuple/list format and direct data format
                self.data = np.array(state[0] if isinstance(state, (tuple, list)) else state)
            return self

        @property
        def r(self):
            """Mimic chumpy's .r property which returns the underlying data"""
            return self.data

