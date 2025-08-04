import sys
import cv2
import numpy as np
import csv
import argparse

def calculate_diameters(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale_factor = 1 / 60  # micrometers per pixel

    diameters = []
    for contour in contours:
        area_in_pixels = cv2.contourArea(contour)
        area_in_micrometers = area_in_pixels * (scale_factor ** 2)
        diameter = 2 * np.sqrt(area_in_micrometers / np.pi)
        diameters.append(diameter)

    return diameters

def save_diameters_csv(filename, image_path, diameters):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Particle Number', 'Diameter (µm)'])
        for i, d in enumerate(diameters, 1):
            writer.writerow([image_path, i, d])
    print(f"Saved diameters to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Calculate nanoparticle diameters from SEM image")
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('-out', '--output', help='Output CSV file to save diameters')
    args = parser.parse_args()

    diameters = calculate_diameters(args.image_path)
    print(f"Diameters (micrometers) of nanoparticles in '{args.image_path}':")
    for i, d in enumerate(diameters, 1):
        print(f"  Particle {i}: {d:.4f}")

    if args.output:
        save_diameters_csv(args.output, args.image_path, diameters)

if __name__ == "__main__":
    main()

