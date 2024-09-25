import os
import shutil

def remove_temp_files(directory):
    """
    Removes temporary files (*.crdownload and *.html) from the given directory and its subdirectories.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.crdownload') or file.endswith('.html'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    print(f"Error deleting temporary file {file_path}: {e}")

def remove_empty_and_no_stl_directories(directory):
    """
    Removes empty directories and directories without .stl files in the given directory tree.
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path) or not any(file.endswith('.stl') for file in os.listdir(dir_path)):
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed directory without .stl files: {dir_path}")
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")

def count_valid_scans(directory):
    """
    Counts the number of subdirectories in the given directory.
    """
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

if __name__ == "__main__":
    # Directory to clean
    DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "antscan_data")

    if not os.path.exists(DOWNLOAD_DIR):
        print(f"The directory {DOWNLOAD_DIR} does not exist.")
    else:
        print(f"Cleaning temporary files in {DOWNLOAD_DIR}...")
        remove_temp_files(DOWNLOAD_DIR)
        remove_empty_and_no_stl_directories(DOWNLOAD_DIR)
        valid_scans = count_valid_scans(DOWNLOAD_DIR)
        print(f"Cleaning complete. Number of valid scans: {valid_scans}")
