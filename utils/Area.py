import sys
import cv2
import numpy as np
import csv
import argparse

def calculate_total_area(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale_factor = 1 / 60  # micrometers per pixel

    total_area = 0
    for contour in contours:
        area_in_pixels = cv2.contourArea(contour)
        area_in_micrometers = area_in_pixels * (scale_factor ** 2)
        total_area += area_in_micrometers

    return total_area

def save_total_area_csv(filename, image_path, total_area):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Total Area (µm²)'])
        writer.writerow([image_path, total_area])
    print(f"Saved total area to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Calculate total nanoparticle area from SEM image")
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('-out', '--output', help='Output CSV file to save the result')
    args = parser.parse_args()

    total_area = calculate_total_area(args.image_path)
    print(f"Total nanoparticle area (micrometers^2) in '{args.image_path}': {total_area}")

    if args.output:
        save_total_area_csv(args.output, args.image_path, total_area)

if __name__ == "__main__":
    main()

