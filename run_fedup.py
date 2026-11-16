"""End-to-end FedUP experiment.

1. Train a FedAvg model in which some clients run a label-flipping or backdoor attack.
2. Once the malicious clients are detected, run FedUP: build the unlearning mask
   from the last-round client models and prune it from the benign average.
3. Run recovery rounds with the benign clients only.

Example:
    python run_fedup.py --dataset cifar10 --model resnet18 --attack backdoor \
        --num-clients 10 --malicious 3 4 7 --distribution iid
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from fedup import (prunable_layer_indices, compute_unlearning_mask, apply_unlearning_mask,
                   benign_similarity, pruning_percentage_from_similarity)
from fedup.data import load_dataset, partition_data, flip_labels, inject_backdoor
from fedup.federated import make_datasets, malicious_accuracy, local_training, run_fedavg
from fedup.models import MODELS

DEFAULT_FLIP_PAIRS = {"cifar10": [(1, 9)], "fashion_mnist": [(4, 6)]}
DEFAULT_FLIP_PAIRS_NON_IID = {"cifar10": [(1, 9), (0, 2), (5, 3)],
                              "fashion_mnist": [(4, 6), (6, 0), (1, 3)]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["cifar10", "fashion_mnist"], default="cifar10")
    p.add_argument("--model", choices=list(MODELS), default=None,
                   help="defaults to the model used in the paper for the chosen dataset/attack")
    p.add_argument("--attack", choices=["label_flipping", "backdoor"], default="label_flipping")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--malicious", type=int, nargs="+", default=[3, 4, 7],
                   help="indices of the malicious clients (assumed detected)")
    p.add_argument("--distribution", choices=["iid", "non_iid"], default="iid")
    p.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration for non_iid")
    p.add_argument("--local-epochs", type=int, default=None,
                   help="defaults to 1 for label flipping and 3 for backdoor")
    p.add_argument("--pruning", type=float, default=None,
                   help="pruning percentage P; if omitted it is derived from the benign similarity")
    p.add_argument("--backdoor-classes", type=int, nargs=2, default=[1, 0], metavar=("SOURCE", "TARGET"))
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.model is None:
        args.model = "resnet18" if args.attack == "backdoor" else f"{args.dataset}_cnn"
    if args.local_epochs is None:
        args.local_epochs = 3 if args.attack == "backdoor" else 1
    if args.attack == "backdoor" and args.dataset != "cifar10":
        raise ValueError("the backdoor trigger is defined for CIFAR-10")

    # --- data and attack -------------------------------------------------------
    (x_train, y_train), test_data = load_dataset(args.dataset)
    local_data, local_labels = partition_data(x_train, y_train, args.num_clients,
                                              args.distribution, args.alpha, rng=rng)
    malicious = sorted(set(args.malicious))
    if args.attack == "label_flipping":
        pairs = (DEFAULT_FLIP_PAIRS if args.distribution == "iid" else DEFAULT_FLIP_PAIRS_NON_IID)[args.dataset]
        poisoned = flip_labels(local_labels, malicious, pairs,
                               max_fraction=None if args.distribution == "iid" else 0.1, rng=rng)
    else:
        poisoned = inject_backdoor(local_data, local_labels, malicious, *args.backdoor_classes)
    datasets = make_datasets(local_data, local_labels)
    eval_mal = lambda m: malicious_accuracy(m, local_data, local_labels, poisoned)

    # --- poisoned federated training --------------------------------------------
    build = MODELS[args.model]
    global_model = build()
    clients = [build() for _ in range(args.num_clients)]
    for c in clients:
        c.set_weights(global_model.get_weights())
    everyone = list(range(args.num_clients))
    benign = [i for i in everyone if i not in malicious]

    t0 = time.time()
    training = run_fedavg(global_model, clients, datasets, everyone, args.local_epochs, test_data,
                          eval_mal, args.patience, args.max_rounds, "Training round")
    before = {"test_acc": training[-1]["test_acc"], "malicious_acc": training[-1]["malicious_acc"]}

    # --- FedUP --------------------------------------------------------------------
    global_weights = global_model.get_weights()
    last_round = local_training(clients, datasets, everyone, args.local_epochs)
    similarity = benign_similarity(last_round, malicious)
    pruning = args.pruning if args.pruning is not None else pruning_percentage_from_similarity(similarity)
    layers = prunable_layer_indices(global_model)
    mask, benign_avg = compute_unlearning_mask(last_round, global_weights, malicious, layers, pruning)
    pruned = apply_unlearning_mask(benign_avg, mask, layers)
    global_model.set_weights(pruned)
    after_pruning = {"test_acc": global_model.evaluate(*test_data, verbose=0)[1],
                     "malicious_acc": eval_mal(global_model)}
    print(f"FedUP: benign similarity {similarity:.4f}, pruning {pruning:.2f}% -> "
          f"test acc {after_pruning['test_acc']:.4f} | malicious acc {after_pruning['malicious_acc']:.4f}")

    # --- recovery with the benign clients ---------------------------------------
    for i in benign:
        clients[i].set_weights(pruned)
    recovery = run_fedavg(global_model, clients, datasets, benign, args.local_epochs, test_data,
                          eval_mal, args.patience, args.max_rounds, "Recovery round")
    best = max(range(len(recovery)), key=lambda r: recovery[r]["test_acc"])

    results = {
        "config": vars(args),
        "before_unlearning": before,
        "after_pruning": after_pruning,
        "after_recovery": recovery[best],
        "recovery_rounds": best + 1,
        "training_rounds": len(training),
        "benign_similarity": similarity,
        "pruning_percentage": pruning,
        "training_history": training,
        "recovery_history": recovery,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(args.out, exist_ok=True)
    name = (f"{args.dataset}_{args.model}_{args.attack}_{args.distribution}_"
            f"{args.num_clients}c_{len(malicious)}m_seed{args.seed}.json")
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps({k: results[k] for k in ("before_unlearning", "after_pruning", "after_recovery",
                                                "recovery_rounds", "pruning_percentage")}, indent=2, default=float))


if __name__ == "__main__":
    main()
