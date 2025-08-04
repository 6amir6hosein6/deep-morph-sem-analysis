import sys
from calculate_score import calculate_score

def calculate_non_adhesion(image_path):
    scores = calculate_score(image_path)
    non_adhesion_index = scores[0]
    return non_adhesion_index

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_non_adhesion_index.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    nai = calculate_non_adhesion(image_path)
    print(f"Non-Adhesion Index for image '{image_path}': {nai}")

if __name__ == "__main__":
    main()

