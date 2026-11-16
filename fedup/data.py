"""Datasets, client partitioning, and the label-flipping and backdoor attacks."""

import numpy as np
import tensorflow as tf

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])

# 3x3 white square (in normalized CIFAR-10 space), placed in the bottom-right corner
BACKDOOR_PATTERN = np.stack([np.full((3, 3), v) for v in (2.5141, 2.5968, 2.7537)])


def load_dataset(name):
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = (x_train / 255.0 - CIFAR10_MEAN) / CIFAR10_STD
        x_test = (x_test / 255.0 - CIFAR10_MEAN) / CIFAR10_STD
    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train / 255.0, -1)
        x_test = np.expand_dims(x_test / 255.0, -1)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return (x_train.astype("float32"), y_train.reshape(-1)), (x_test.astype("float32"), y_test.reshape(-1))


def partition_data(x, y, num_clients, mode="iid", alpha=1.0, num_classes=10, rng=np.random):
    """Split the training set among clients: equal contiguous shards ('iid') or
    a per-class Dirichlet split ('non_iid')."""
    if mode == "iid":
        size = len(x) // num_clients
        return ([x[i * size:(i + 1) * size].copy() for i in range(num_clients)],
                [y[i * size:(i + 1) * size].copy() for i in range(num_clients)])
    if mode != "non_iid":
        raise ValueError(f"Unknown distribution: {mode}")
    client_idx = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = rng.permutation(np.where(y == c)[0])
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for client, part in enumerate(np.split(idx, cuts)):
            client_idx[client].extend(part)
    return [x[i].copy() for i in client_idx], [y[i].copy() for i in client_idx]


def flip_labels(local_labels, malicious_clients, flip_pairs, max_fraction=None, rng=np.random):
    """Label-flipping attack. Returns the indices of the flipped samples of each
    malicious client (used to measure the accuracy on malicious data)."""
    poisoned = {}
    for c in malicious_clients:
        budget = None if max_fraction is None else int(len(local_labels[c]) * max_fraction)
        flipped = []
        for src, dst in flip_pairs:
            idx = np.where(local_labels[c] == src)[0]
            if budget is not None:
                idx = rng.choice(idx, min(len(idx), budget - len(flipped)), replace=False)
            local_labels[c][idx] = dst
            flipped.extend(idx)
            if budget is not None and len(flipped) >= budget:
                break
        poisoned[c] = np.array(flipped, dtype=int)
    return poisoned


def inject_backdoor(local_data, local_labels, malicious_clients, source_class, target_class,
                    pattern=BACKDOOR_PATTERN):
    """Backdoor attack: stamp the trigger on the malicious clients' samples of
    source_class and relabel them as target_class."""
    h, w = pattern.shape[1], pattern.shape[2]
    poisoned = {}
    for c in malicious_clients:
        idx = np.where(local_labels[c] == source_class)[0]
        local_data[c][idx, -h:, -w:, :] = pattern
        local_labels[c][idx] = target_class
        poisoned[c] = idx
    return poisoned
