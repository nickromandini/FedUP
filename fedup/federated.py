"""Minimal FedAvg simulation used to train the poisoned model and to run the
recovery rounds after unlearning."""

import numpy as np
import tensorflow as tf

from .unlearning import average_weights


def make_datasets(local_data, local_labels, batch_size=32):
    return [tf.data.Dataset.from_tensor_slices((x, y)).cache().shuffle(len(x))
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
            for x, y in zip(local_data, local_labels)]


def malicious_accuracy(model, local_data, local_labels, poisoned):
    """Mean accuracy of the model on the poisoned samples of the malicious clients."""
    accs = [model.evaluate(local_data[c][idx], local_labels[c][idx], verbose=0)[1]
            for c, idx in poisoned.items() if len(idx)]
    return float(np.mean(accs)) if accs else float("nan")


def local_training(clients, datasets, participants, epochs):
    for i in participants:
        clients[i].fit(datasets[i], epochs=epochs, verbose=0)
    return [clients[i].get_weights() for i in participants]


def run_fedavg(global_model, clients, datasets, participants, epochs, test_data,
               eval_fn=None, patience=10, max_rounds=200, log_prefix="Round"):
    """FedAvg with early stopping on test accuracy. Returns the history and the
    number of rounds run. Clients keep their own optimizer state across rounds."""
    x_test, y_test = test_data
    history, best, wait = [], -1.0, patience
    for rnd in range(max_rounds):
        weights = local_training(clients, datasets, participants, epochs)
        global_weights = average_weights(weights)
        global_model.set_weights(global_weights)
        acc = global_model.evaluate(x_test, y_test, verbose=0)[1]
        mal = eval_fn(global_model) if eval_fn else float("nan")
        history.append({"round": rnd, "test_acc": acc, "malicious_acc": mal})
        print(f"{log_prefix} {rnd}: test acc {acc:.4f} | malicious acc {mal:.4f}")
        for i in participants:
            clients[i].set_weights(global_weights)
        if acc > best:
            best, wait = acc, patience
        else:
            wait -= 1
            if wait == 0:
                break
    return history
