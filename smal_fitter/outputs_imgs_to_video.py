import cv2
import os


def create_video_from_images(image_folder, video_name, frame_rate):
    # Get list of all PNG files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # sort images by time to write
    images.sort(key=lambda img: os.path.getmtime(os.path.join(image_folder, img)))

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get the size
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video {video_name} created successfully!")


# Parameters
image_folder = '/home/fabi/dev/SMILify/checkpoints/20240807-155427/SMIL_07_synth'  # Replace with the path to your folder containing PNGs
video_name = "SMIL-fit-" + image_folder.split("/")[-1] + '-output_video.mp4'  # The name of the output video file
frame_rate = 25  # You can adjust the frame rate

# Create video from images
create_video_from_images(image_folder, video_name, frame_rate)
