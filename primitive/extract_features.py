import cv2
import numpy as np

from pathlib import Path


def compute_lbp(gray_image: np.ndarray) -> np.ndarray:
    """
    Compute basic 3x3 Local Binary Pattern for a grayscale image.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image with shape (H, W).

    Returns
    -------
    lbp : np.ndarray
        LBP image with shape (H-2, W-2), values from 0 to 255.
    """

    if len(gray_image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    gray = gray_image.astype(np.uint8)

    height, width = gray.shape
    lbp = np.zeros((height - 2, width - 2), dtype=np.uint8)

    # Neighbor offsets in clockwise order
    neighbors = [
        (-1, -1),  # top-left
        (-1, 0),   # top
        (-1, 1),   # top-right
        (0, 1),    # right
        (1, 1),    # bottom-right
        (1, 0),    # bottom
        (1, -1),   # bottom-left
        (0, -1),   # left
    ]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = gray[y, x]
            binary_value = 0

            for i, (dy, dx) in enumerate(neighbors):
                neighbor = gray[y + dy, x + dx]

                if neighbor >= center:
                    binary_value |= (1 << (7 - i))

            lbp[y - 1, x - 1] = binary_value

    return lbp


def lbp_histogram(lbp_image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert an LBP image into a 256-bin histogram feature vector.
    """

    hist, _ = np.histogram(
        lbp_image.ravel(),
        bins=256,
        range=(0, 256)
    )

    hist = hist.astype(np.float32)

    if normalize:
        hist /= hist.sum() + 1e-8

    return hist

def extract_lbp_feature(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional but useful: resize all images to same size
    gray = cv2.resize(gray, (256, 256))

    lbp = compute_lbp(gray)
    hist = lbp_histogram(lbp)

    return hist

if __name__ == "__main__":
    image_folder = Path("data")
    output_file = "primitive/lbp_features.npz"

    features = []
    image_paths = []

    for path in image_folder.glob("*.jpg"):
        try:
            feature = extract_lbp_feature(str(path))
        except ValueError as e:
            print(f"Error processing {path}: {e}")
            continue

        features.append(feature)
        image_paths.append(path)

    features = np.vstack(features)

    np.savez(
        output_file,
        features=features,
        image_paths=np.array(image_paths)
    )
