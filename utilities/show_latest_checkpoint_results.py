import pickle as pkl

import os


def find_latest_pkl_file(directory):
    # Checks for the latest pkl file in a given directory
    latest_pkl_file = None
    latest_time = 0

    # Traverse the directory tree
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                file_creation_time = os.path.getctime(file_path)

                # Check if this is the latest file
                if file_creation_time > latest_time:
                    latest_time = file_creation_time
                    latest_pkl_file = file_path

    return latest_pkl_file

directory = "checkpoints/"  # Replace with your directory path
latest_file = find_latest_pkl_file(directory)
if latest_file:
    print(f"The latest .pkl file is: {latest_file}")
else:
    print("No .pkl files found.")

with open(latest_file, 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    stage_pkl_content = u.load()

for key in stage_pkl_content:
    print(key)
    print(stage_pkl_content[key].shape)
    print(stage_pkl_content[key])
