import torch
from tqdm import tqdm

def normalize_embeddings(embeddings) :
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def community_detection(embeddings, threshold=0.6,
                        min_community_size=10, batch_size=1024, show_progress_bar=False):

    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)
    embeddings = normalize_embeddings(embeddings)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in tqdm(
        range(0, len(embeddings), batch_size), desc="Finding clusters", disable=not show_progress_bar
    ):
        # Compute cosine similarity scores
        cos_scores = embeddings[start_idx : start_idx + batch_size] @ embeddings.T

        # Use a torch-heavy approach if the embeddings are on CUDA, otherwise a loop-heavy one
        if embeddings.device.type in ["cuda", "npu"]:
            # Threshold the cos scores and determine how many close embeddings exist per embedding
            threshold_mask = cos_scores >= threshold
            row_wise_count = threshold_mask.sum(1)

            # Only consider embeddings with enough close other embeddings
            large_enough_mask = row_wise_count >= min_community_size
            if not large_enough_mask.any():
                continue

            row_wise_count = row_wise_count[large_enough_mask]
            cos_scores = cos_scores[large_enough_mask]

            # The max is the largest potential community, so we use that in topk
            k = row_wise_count.max()
            _, top_k_indices = cos_scores.topk(k=k, largest=True)

            # Use the row-wise count to slice the indices
            for count, indices in zip(row_wise_count, top_k_indices):
                extracted_communities.append(indices[:count].tolist())
        else:
            # Minimum size for a community
            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

            # Filter for rows >= min_threshold
            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    # Only check top k most similar entries
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    # Check if we need to increase sort_max_size
                    while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                        sort_max_size = min(2 * sort_max_size, len(embeddings))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    extracted_communities.append(top_idx_large[top_val_large >= threshold].tolist())

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities
