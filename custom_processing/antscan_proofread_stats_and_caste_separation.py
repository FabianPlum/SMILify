import os
import json
import shutil
from collections import Counter

def process_obj_files(input_dir, lookup_dir, output_dir):
    # Check if the directories exist
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    if not os.path.isdir(lookup_dir):
        print(f"Error: Lookup directory '{lookup_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all .obj files in the input directory
    obj_files = [f for f in os.listdir(input_dir) if f.lower().endswith('_processed.obj')]

    caste_counter = Counter()

    for filename in obj_files:
        # Strip away "_processed.obj" from the end of the name
        base_name = filename.rsplit('_processed.obj', 1)[0]
        
        # Look for a folder with the base name in the lookup directory
        folder_path = os.path.join(lookup_dir, base_name)
        
        if os.path.isdir(folder_path):
            # Find the JSON file with the base name
            json_files = [f for f in os.listdir(folder_path) if f.startswith(base_name) and f.endswith('.json')]
            
            if json_files:
                json_path = os.path.join(folder_path, json_files[0])
                
                # Load and process the JSON file
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    # Count caste entries and copy files
                    if 'caste' in data:
                        caste = data['caste']
                        caste_counter[caste] += 1
                        
                        # Create caste folder in output directory
                        caste_folder = os.path.join(output_dir, caste)
                        os.makedirs(caste_folder, exist_ok=True)
                        
                        # Copy the input file to the caste folder
                        src_file = os.path.join(input_dir, filename)
                        dst_file = os.path.join(caste_folder, filename)
                        shutil.copy2(src_file, dst_file)
            else:
                print(f"No JSON file found for {base_name} in {folder_path}")
        else:
            print(f"No folder found for {base_name} in {lookup_dir}")

    # Print the list of castes and their counts
    print("Castes and their occurrences:")
    for caste, count in sorted(caste_counter.items()):
        print(f"{caste}: {count}")

    # Print total number of processed files
    print(f"\nTotal number of processed files: {len(obj_files)}")

# Example usage
input_directory = "custom_processing/antscan_proofread"
lookup_directory = "custom_processing/antscan_data"
output_directory = "custom_processing/antscan_proofread_castes"

process_obj_files(input_directory, lookup_directory, output_directory)
