import cv2
import os
import argparse
import numpy as np
from pathlib import Path
import imageio


def convert_images_to_video(input_folder, output_file, scale_factor=1.0, fps=30, as_gif=False, loop=0):
    """
    Convert all images in a folder to a video file.
    
    Args:
        input_folder (str): Path to folder containing images
        output_file (str): Path to output video file
        scale_factor (float): Scale factor to resize images (default: 1.0)
        fps (int): Frames per second for output video (default: 30)
        as_gif (bool): Whether to save as GIF instead of MP4 (default: False)
        loop (int): Number of times to loop the GIF (0 = infinite loop, default)
    """
    # Get all image files from the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(input_folder).glob(f'*{ext.upper()}')))
    
    # Sort the files to ensure proper sequence
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Failed to read image: {image_files[0]}")
        return
    
    # Calculate new dimensions
    h, w = first_image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    if as_gif:
        # For GIF output, use imageio
        frames = []
        print(f"Processing {len(image_files)} images for GIF...")
        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            # Convert from BGR to RGB for GIF
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if scale_factor != 1.0:
                img = cv2.resize(img, (new_w, new_h))
            
            frames.append(img)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        # Save as GIF
        print(f"Saving GIF to {output_file}...")
        imageio.mimsave(output_file, frames, fps=fps, loop=loop)
        print(f"GIF saved to {output_file}")
    else:
        # For MP4 output, use OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (new_w, new_h))
        
        # Process each image
        print(f"Processing {len(image_files)} images for video...")
        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            # Resize if needed
            if scale_factor != 1.0:
                img = cv2.resize(img, (new_w, new_h))
            
            # Write to video
            video_writer.write(img)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        # Release resources
        video_writer.release()
        print(f"Video saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert images in a folder to a video file')
    parser.add_argument('input_folder', type=str, help='Path to folder containing images')
    parser.add_argument('--output', type=str, help='Path to output video file (default: output.mp4)')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for image resizing (default: 1.0)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--gif', action='store_true', help='Save as GIF instead of MP4')
    parser.add_argument('--loop', type=int, default=0, 
                        help='Number of times to loop the GIF (0 = infinite loop, default)')
    
    args = parser.parse_args()
    
    # Set default output file if not specified, with appropriate extension
    if args.output:
        output_file = args.output
    else:
        extension = '.gif' if args.gif else '.mp4'
        output_file = os.path.join(os.getcwd(), f'output{extension}')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    convert_images_to_video(args.input_folder, output_file, args.scale, args.fps, args.gif, args.loop)


if __name__ == "__main__":
    main()
