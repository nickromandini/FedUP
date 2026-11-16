# FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks

Official implementation of **FedUP**, a lightweight federated unlearning (FU) algorithm that
removes the influence of malicious clients from a poisoned federated learning (FL) model by
pruning the connections they most affected.

> N. Romandini, C. Borcea, R. Montanari, L. Foschini.
> *FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks.*
> [arXiv:2508.13853](https://arxiv.org/abs/2508.13853)

## How it works

Once the malicious clients are detected, the server uses only the client models received in
the **last training round**. No history, auxiliary data, or client cooperation is needed.

1. Average the benign models and the malicious models into two reference models.
2. Compute their squared difference and scale it by the magnitude of the global weights.
3. In every dense and convolutional layer, select the top-P% weights. These form the
   unlearning mask.
4. Zero the masked weights in the benign average.
5. Run a few recovery rounds with the benign clients only.

The pruning percentage P can be set by hand or derived from the similarity of the benign
updates. The mean pairwise cosine similarity of the benign clients' last-layer weights
`SIM` is normalized to `z = (SIM - 0.5) / 0.5`, and then
`P = (P_max - P_min) * z^gamma + P_min`, with `P_min = 1%`, `P_max = 15%` and `gamma = 5`.

## Repository structure

```
fedup/
  unlearning.py   # FedUP: mask generation, mask application, choice of P
  federated.py    # minimal FedAvg simulation (training and recovery rounds)
  data.py         # datasets, IID/Dirichlet partitioning, label-flipping and backdoor attacks
  models.py       # CNNs for CIFAR-10 / Fashion-MNIST and ResNet-18
run_fedup.py      # end-to-end experiment: poisoned training -> FedUP -> recovery
```

## Installation

```bash
git clone https://github.com/nickromandini/FedUP.git
cd FedUP
pip install -r requirements.txt
```

The code uses TensorFlow 2 with Keras 3 (TensorFlow 2.16 or newer). Datasets are
downloaded automatically through `tf.keras.datasets`.

## Usage

Label-flipping attack on CIFAR-10 with 10 clients, 3 of which are malicious:

```bash
python run_fedup.py --dataset cifar10 --attack label_flipping \
    --num-clients 10 --malicious 3 4 7 --distribution iid
```

Backdoor attack on CIFAR-10 with ResNet-18 and non-IID data:

```bash
python run_fedup.py --dataset cifar10 --attack backdoor \
    --num-clients 20 --malicious 3 4 7 8 11 12 --distribution non_iid --pruning 3
```

Main options:

| Option | Description |
|---|---|
| `--dataset` | `cifar10` or `fashion_mnist` |
| `--attack` | `label_flipping` or `backdoor` (3x3 trigger, class 1 -> 0) |
| `--model` | `cifar10_cnn`, `fashion_mnist_cnn` or `resnet18` (defaults to the paper's model) |
| `--num-clients`, `--malicious` | number of clients and indices of the detected malicious clients |
| `--distribution` | `iid` or `non_iid` (Dirichlet, `--alpha 1.0`) |
| `--pruning` | pruning percentage P; if omitted, it is derived from the benign similarity |
| `--local-epochs` | local epochs per round (default 1 for label flipping, 3 for backdoor) |
| `--patience`, `--max-rounds` | early stopping of training and recovery |

The script prints the test accuracy and the accuracy on the malicious data before
unlearning, right after pruning, and after recovery, together with the number of recovery
rounds. The full history is saved as JSON in `results/`.

### Using FedUP in your own FL code

```python
from fedup import (prunable_layer_indices, compute_unlearning_mask,
                   apply_unlearning_mask, benign_similarity,
                   pruning_percentage_from_similarity)

# client_weights: list of model.get_weights() received in the last round
# global_weights: global model weights of the previous round
p = pruning_percentage_from_similarity(benign_similarity(client_weights, malicious))
layers = prunable_layer_indices(global_model)
mask, benign_avg = compute_unlearning_mask(client_weights, global_weights, malicious, layers, p)
global_model.set_weights(apply_unlearning_mask(benign_avg, mask, layers))
# ...then continue FedAvg with the benign clients only
```

## Citation

If you use this code, please cite:

```bibtex
@article{romandini2025fedup,
  title   = {FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks},
  author  = {Romandini, Nicol{\`o} and Borcea, Cristian and Montanari, Rebecca and Foschini, Luca},
  journal = {arXiv preprint arXiv:2508.13853},
  year    = {2025}
}
```
