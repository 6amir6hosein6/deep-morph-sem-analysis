import sys
from calculate_score import calculate_score

def calculate_scattering(image_path):
    scores = calculate_score(image_path)
    scattering_index = scores[1]
    return scattering_index

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_scattering_index.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    si = calculate_scattering(image_path)
    print(f"Scattering Index for image '{image_path}': {si}")

if __name__ == "__main__":
    main()

