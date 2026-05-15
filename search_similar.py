import random

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_lbp_features(feature_file: str = "lbp_features.npz"):
    """
    Load saved LBP features and image paths.

    Returns
    -------
    features : np.ndarray
        Feature matrix with shape (N, 256)

    image_paths : np.ndarray
        Array of image paths with shape (N,)
    """

    data = np.load(feature_file, allow_pickle=True)

    features = data["features"]
    image_paths = data["image_paths"]

    return features, image_paths


def similarity_search_by_index(
    query_index: int,
    features: np.ndarray,
    image_paths: np.ndarray,
    top_k: int = 5
):
    """
    Find the most similar images to an image already in the dataset.
    """

    if query_index < 0 or query_index >= len(features):
        raise ValueError("query_index is out of range.")

    query_feature = features[query_index].reshape(1, -1)

    similarities = cosine_similarity(query_feature, features)[0]

    # Sort from most similar to least similar
    sorted_indices = np.argsort(similarities)[::-1]

    # Remove the query image itself
    sorted_indices = [i for i in sorted_indices if i != query_index and similarities[i] != 1.0]

    results = []

    for i in sorted_indices[:top_k]:
        results.append({
            "image_path": image_paths[i],
            "similarity": float(similarities[i])
        })

    return results


def print_results(query_index, image_paths, results):
    print("Query image:")
    print(image_paths[query_index])
    print()

    print("Most similar images:")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['image_path']}")
        print(f"   Similarity: {result['similarity']:.4f}")


if __name__ == "__main__":
    features, image_paths = load_lbp_features("primitive/lbp_features.npz")

    query_index = random.randint(0, len(features) - 1)
    top_k = 5

    results = similarity_search_by_index(
        query_index=query_index,
        features=features,
        image_paths=image_paths,
        top_k=top_k
    )

    print_results(query_index, image_paths, results)
