from pathlib import Path
import pickle

import cv2 as cv
from tqdm import tqdm


IMAGES_FOLDER = Path("data/images")
FEATURES_PATH = Path("data/features/sift_descriptors.pkl")


def extract_sift_descriptors(image_path: Path, sift) -> tuple[str, object] | None:
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Skipping unreadable image: {image_path}")
        return None

    keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        print(f"Skipping image without descriptors: {image_path}")
        return None

    return image_path.name, descriptors


def main() -> None:
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    sift = cv.SIFT_create()
    image_paths = list(IMAGES_FOLDER.glob("*.jpg"))

    features = {}

    for image_path in tqdm(image_paths, desc="Extracting SIFT features"):
        result = extract_sift_descriptors(image_path, sift)

        if result is None:
            continue

        filename, descriptors = result
        features[filename] = descriptors

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved SIFT descriptors for {len(features)} images to {FEATURES_PATH}")


if __name__ == "__main__":
    main()