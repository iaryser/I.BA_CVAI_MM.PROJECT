from pathlib import Path
import pickle

import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm


QUERY_IMAGE = Path("data/example_images/apartment_4001732069_3.jpg")
FEATURES_PATH = Path("data/features/sift_descriptors.pkl")
IMAGES_FOLDER = Path("data/images")


def calculate_similarity(query_descriptors, image_descriptors, matcher) -> float:
    matches = matcher.knnMatch(query_descriptors, image_descriptors, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    normalizer = min(len(query_descriptors), len(image_descriptors))

    if normalizer == 0:
        return 0.0

    return len(good_matches) / normalizer


def plot_top_results(query_image_path: Path, top_results: list[tuple[str, float]]) -> None:
    query_img = cv.imread(str(query_image_path))

    if query_img is None:
        raise FileNotFoundError(f"Could not read query image: {query_image_path.resolve()}")

    query_img = cv.cvtColor(query_img, cv.COLOR_BGR2RGB)

    total_images = len(top_results) + 1

    plt.figure(figsize=(4 * total_images, 5))

    plt.subplot(1, total_images, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")

    for index, (filename, score) in enumerate(top_results, start=2):
        image_path = IMAGES_FOLDER / filename

        img = cv.imread(str(image_path))

        if img is None:
            print(f"Could not plot unreadable image: {image_path}")
            continue

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        plt.subplot(1, total_images, index)
        plt.imshow(img)
        plt.title(f"{filename}\nScore: {score:.4f}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    query_img = cv.imread(str(QUERY_IMAGE), cv.IMREAD_GRAYSCALE)

    if query_img is None:
        raise FileNotFoundError(f"Could not read query image: {QUERY_IMAGE.resolve()}")

    sift = cv.SIFT_create()
    matcher = cv.BFMatcher()

    query_keypoints, query_descriptors = sift.detectAndCompute(query_img, None)

    if query_descriptors is None:
        raise ValueError("No SIFT descriptors found in query image.")

    with open(FEATURES_PATH, "rb") as f:
        database_features = pickle.load(f)

    similarity_scores = {}

    for filename, image_descriptors in tqdm(database_features.items(), desc="Matching images"):
        similarity = calculate_similarity(
            query_descriptors=query_descriptors,
            image_descriptors=image_descriptors,
            matcher=matcher,
        )

        similarity_scores[filename] = similarity

    top_5 = sorted(
        similarity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    print("\nTop 5 most similar images:")
    for filename, score in top_5:
        print(f"{filename}: {score:.4f}")

    plot_top_results(QUERY_IMAGE, top_5)


if __name__ == "__main__":
    main()