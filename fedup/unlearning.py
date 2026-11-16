"""Core FedUP algorithm (Algorithms 1 and 2 of the paper).

All functions operate on lists of NumPy arrays, i.e. the output of
``tf.keras.Model.get_weights()``, so they are framework-agnostic.
"""

import numpy as np
from scipy.spatial.distance import pdist


def average_weights(weights_list):
    """FedAvg (unweighted) of a list of per-client weight lists."""
    return [np.mean(layer, axis=0) for layer in zip(*weights_list)]


def prunable_layer_indices(model):
    """Indices (in ``model.get_weights()`` order) of the dense and convolutional
    weight tensors, kernels and biases. Only these tensors are pruned; normalization
    layers are left untouched."""
    names = [f"{layer.name} {w.path}" for layer in model.layers for w in layer.weights]
    return [i for i, n in enumerate(names) if "dense" in n or "conv" in n]


def compute_unlearning_mask(client_weights, global_weights, malicious_clients,
                            layers_indices, pruning_percentage):
    """Generation of the unlearning mask (Algorithm 1).

    Args:
        client_weights: list of per-client weight lists received in the last round.
        global_weights: weights of the global model of the previous round.
        malicious_clients: indices of the clients flagged by the detector.
        layers_indices: indices of the prunable tensors (see ``prunable_layer_indices``).
        pruning_percentage: percentage P of weights pruned in each prunable tensor.

    Returns:
        (mask, benign_average): a list with one boolean array per tensor (``False``
        for non-prunable tensors) and the average of the benign client models.
    """
    benign = [w for i, w in enumerate(client_weights) if i not in malicious_clients]
    malicious = [w for i, w in enumerate(client_weights) if i in malicious_clients]
    benign_avg = average_weights(benign)
    malicious_avg = average_weights(malicious)

    mask = []
    for idx, (m, b) in enumerate(zip(malicious_avg, benign_avg)):
        if idx not in layers_indices:
            mask.append(False)
            continue
        # squared difference between the reference models, scaled by the
        # magnitude of the corresponding global weights
        rank = (m - b) ** 2 * np.abs(global_weights[idx])
        threshold = np.percentile(rank, 100 - pruning_percentage)
        mask.append(rank > threshold)
    return mask, benign_avg


def apply_unlearning_mask(benign_average, mask, layers_indices):
    """Zero out the selected weights of the benign average (Algorithm 2)."""
    pruned = [w.copy() for w in benign_average]
    for idx in layers_indices:
        pruned[idx][mask[idx]] = 0
    return pruned


def benign_similarity(client_weights, malicious_clients):
    """Mean pairwise (centered) cosine similarity among the benign clients'
    last-layer weights, used to choose the pruning percentage."""
    vectors = np.array([w[-1].flatten() for i, w in enumerate(client_weights)
                        if i not in malicious_clients])
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    return float(np.mean(1 - pdist(centered, metric="cosine")))


def pruning_percentage_from_similarity(similarity, p_min=0.01, p_max=0.15, gamma=5,
                                       sim_min=0.5, sim_max=1.0):
    """Practical recipe of the paper: normalize the benign similarity to
    z in [0, 1] and map it to P = (p_max - p_min) * z^gamma + p_min.
    Returns P as a percentage."""
    z = float(np.clip((similarity - sim_min) / (sim_max - sim_min), 0.0, 1.0))
    return 100 * ((p_max - p_min) * z ** gamma + p_min)
