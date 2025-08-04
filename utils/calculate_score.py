import cv2
import numpy as np

# Choose metric type for size measurement ('Area' or 'Diameter')
T = 'Area'

def calculate_score(image_path):
    """
    Calculate Non-Adhesion Index (NAI), Scattering Index (SI), and particle count from an SEM mask image.

    Parameters:
        image_path (str): Path to the grayscale SEM mask image.

    Returns:
        list: [Non-Adhesion Index, Scattering Index, Number of particles]
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Find contours of particles in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale factor: micrometers per pixel (adjust according to your imaging system)
    scale_factor = 1 / 60  # micrometers per pixel

    # Calculate particle areas or diameters in micrometers
    particle_sizes = []
    for contour in contours:
        area_in_pixels = cv2.contourArea(contour)
        area_in_micrometers = area_in_pixels * (scale_factor ** 2)
        

        particle_sizes.append(area_in_micrometers)

    # Non-Adhesion Index: number of particles divided by total area
    if sum(particle_sizes) == 0:
        param_one = 0
    else:
        param_one = len(particle_sizes) / sum(particle_sizes)

    # Calculate scattering index: average distance between particle centers
    total_distance = 0.0
    num_particles = len(contours)

    if num_particles <= 1:
        param_two = 0
    else:
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                center_i = np.mean(contours[i], axis=0)[0]
                center_j = np.mean(contours[j], axis=0)[0]
                distance = np.linalg.norm(center_i - center_j)
                total_distance += distance

        param_two = total_distance / num_particles

    return [param_one, param_two, num_particles]


