import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load JSON file
with open('data/replicAnt_trials/TEST_ANGLES/TEST_ANGLES_27.json', 'r') as file:
    data = json.load(file)


# Function to convert euler angles to unit vector
def euler_to_unit_vector(euler_angles):
    try:
        pitch, yaw, roll = euler_angles['x'], euler_angles['y'], euler_angles['z']
    except:
        roll, pitch, yaw = -euler_angles['pitch'], -euler_angles['yaw'], -euler_angles['roll']



    # Compute the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])

    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])

    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]])

    # The resulting unit vector (assuming it's along the z-axis in the local frame)
    unit_vector = np.dot(R_z, np.dot(R_y, R_x))
    return unit_vector


# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract and plot the data
keypoints = data['iterationData']['subject Data'][0]["1"]['keypoints']

for key, value in keypoints.items():
    # Get the 3D position
    position = np.array([value['3DPos']['x'], value['3DPos']['y'], value['3DPos']['z']])

    # Get the unit vector corresponding to the euler angles
    unit_vector = euler_to_unit_vector(value['eulerAngles'])

    unit_vector = [value['eulerAngles']['x'],
                   value['eulerAngles']['y'],
                   value['eulerAngles']['z']]

    # Plot the point
    ax.scatter(-position[1], position[2], position[0], color='blue')

    """
    # Plot the vector
    ax.quiver(-position[1], position[2], position[0],
              unit_vector[0], unit_vector[1], unit_vector[2],
              length=5, color='red')
    """

# Plot the camera
camera_translation = [data['iterationData']['camera']["Location"]["x"],
                      data['iterationData']['camera']["Location"]["y"],
                      data['iterationData']['camera']["Location"]["z"]]

ax.scatter(-camera_translation[1], camera_translation[2], camera_translation[0], color='green')

# Get the unit vector corresponding to the euler angles
camera_vector = euler_to_unit_vector(data['iterationData']['camera']["Rotation"])

unit_vector = [value['eulerAngles']['x'],
               value['eulerAngles']['y'],
               value['eulerAngles']['z']]


# Plot the vector
ax.quiver(-camera_translation[1], camera_translation[2], camera_translation[0],
          unit_vector[0], unit_vector[1], unit_vector[2],
          length=5, color='red')


# Set labels and show plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
