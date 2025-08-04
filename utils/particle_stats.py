import numpy as np
import cv2
import pandas as pd

# Choose analysis type: 'Area' or 'Diameter'
T = 'Area'  # or 'Diameter'

# Load the black-and-white segmented image
image_path = 'Detected-Maski.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Find contours in the image
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define scale (2 µm = 120 pixels) → 1 pixel = 1/60 µm
scale_factor = 1 / 60  # µm per pixel

values = []
for contour in contours:
    area_in_pixels = cv2.contourArea(contour)
    area_in_um2 = area_in_pixels * (scale_factor ** 2)
    diameter_in_um = 2 * np.sqrt(area_in_um2 / np.pi)
    
    if T == 'Area' and area_in_um2 > 0:
        values.append(area_in_um2)
    elif T == 'Diameter' and diameter_in_um > 0:
        values.append(diameter_in_um * 1000)  # Convert to nanometers

# Save to Excel
df = pd.DataFrame(values, columns=[f'{T}'])
df.to_excel('output.xlsx', index=False)

# Print summary
print(f"\n===== Particle {T} Statistics =====")
print(f"Number of particles  : {len(values)}")
print(f"Minimum {T}          : {np.min(values):.4f}")
print(f"Maximum {T}          : {np.max(values):.4f}")
print(f"Mean {T}             : {np.mean(values):.4f}")

# Image area stats
image_width, image_height = image.shape[1], image.shape[0]
total_area_px = image_width * image_height
total_area_um2 = total_area_px * (scale_factor ** 2)
particle_area_um2 = sum(values) if T == 'Area' else np.nan  # Only valid if computing area

print("\n===== Image Area Info =====")
print(f"Total image area (µm²)        : {total_area_um2:.2f}")
if T == 'Area':
    print(f"Total particles area (µm²)    : {particle_area_um2:.2f}")
    print(f"Particles-to-image ratio (%)  : {particle_area_um2 / total_area_um2 * 100:.2f}%")

