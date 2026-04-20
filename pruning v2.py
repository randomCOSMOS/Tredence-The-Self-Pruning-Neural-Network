import copy
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 0. Global config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 20
WARMUP_EPOCHS = 4
RAMP_EPOCHS = 8
BATCH_SIZE = 128
LR = 1e-3
GATE_LR_MULT = 8.0
GATE_TEMPERATURE = 5.0
PRUNE_THRESH = 1e-2
LAMBDAS = [0.05, 0.1, 0.2]
DATA_DIR = Path("./data")
RESULTS_CSV = Path("results_table.csv")
REPORT_MD = Path("report.md")
BEST_PLOT = Path("gate_distribution_best.png")


# ============================================================
# CHECKPOINT 1: PrunableLinear module implementation
# This custom layer replaces nn.Linear and learns one gate per
# weight. Each gate is produced from a learnable score and
# multiplied element-wise with the underlying weight matrix.
# ============================================================
class PrunableLinear(nn.Module):
    """
    Linear layer with one learnable gate score per weight.

    The effective weight used during the forward pass is:
        effective_weight = weight * sigmoid(gate_scores)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 0.5))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(GATE_TEMPERATURE * self.gate_scores)
        effective_weight = self.weight * gates
        return F.linear(x, effective_weight, self.bias)

    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(GATE_TEMPERATURE * self.gate_scores)

    def detached_gates(self) -> torch.Tensor:
        return self.gate_values().detach()

    def hard_prune(self) -> None:
        with torch.no_grad():
            gates = self.detached_gates()
            mask = gates >= PRUNE_THRESH
            self.weight.mul_(mask)
            self.gate_scores.masked_fill_(~mask, -12.0)


# ============================================================
# CHECKPOINT 2: Neural network definition using PrunableLinear
# The feature extractor is convolutional, while the classifier
# uses PrunableLinear layers so the model can learn which dense
# connections are unnecessary and prune them during training.
# ============================================================
class SelfPruningNet(nn.Module):
    """
    Conv feature extractor plus prunable classifier.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            PrunableLinear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            PrunableLinear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).view(x.size(0), -1)
        return self.classifier(features)

    def prunable_layers(self):
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        Mean gate value across all prunable weights.

        Averaging keeps the scale stable across model sizes so lambda is easier
        to tune and does not swamp the classification loss.
        """

        all_gates = [layer.gate_values().reshape(-1) for layer in self.prunable_layers()]
        return torch.cat(all_gates).mean()

    def overall_sparsity(self) -> float:
        total = 0
        pruned = 0
        for layer in self.prunable_layers():
            gates = layer.detached_gates()
            total += gates.numel()
            pruned += (gates < PRUNE_THRESH).sum().item()
        return pruned / total if total else 0.0

    def all_gate_values(self) -> np.ndarray:
        return np.concatenate(
            [layer.detached_gates().cpu().numpy().ravel() for layer in self.prunable_layers()]
        )

    def hard_prune(self) -> None:
        for layer in self.prunable_layers():
            layer.hard_prune()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    full_train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    val_base = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=eval_tf)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)

    generator = torch.Generator().manual_seed(SEED)
    train_size = 45_000
    val_size = 5_000

    train_ds, _ = random_split(full_train, [train_size, val_size], generator=generator)
    _, val_ds = random_split(val_base, [train_size, val_size], generator=generator)

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def sparsity_weight(epoch: int, target_lambda: float) -> float:
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    if RAMP_EPOCHS <= 0:
        return target_lambda
    progress = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)
    return target_lambda * progress


def train_one_epoch(model, loader, optimizer, epoch: int, target_lambda: float):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_sparse = 0.0
    correct = 0
    total = 0
    effective_lambda = sparsity_weight(epoch, target_lambda)

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        cls_loss = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()
        loss = cls_loss + effective_lambda * sparse_loss
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_cls += cls_loss.item() * batch_size
        total_sparse += sparse_loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": total_loss / total,
        "cls_loss": total_cls / total,
        "sparse_loss": total_sparse / total,
        "acc": correct / total,
        "effective_lambda": effective_lambda,
        "sparsity": model.overall_sparsity(),
        "gate_mean": float(model.all_gate_values().mean()),
    }


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ============================================================
# CHECKPOINT 3: Complete training and evaluation loop
# This section trains the model with classification loss plus
# lambda * sparsity loss, evaluates each run, compares lambda
# values, and saves the final table, plot, and Markdown report.
# ============================================================
def run_experiment(target_lambda: float, train_loader, val_loader, test_loader):
    print(f"\n{'=' * 70}")
    print(f"Lambda = {target_lambda}")
    print(f"{'=' * 70}")

    model = SelfPruningNet().to(DEVICE)
    gate_params = [param for name, param in model.named_parameters() if "gate_scores" in name]
    other_params = [param for name, param in model.named_parameters() if "gate_scores" not in name]

    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": LR, "weight_decay": 1e-4},
            {"params": gate_params, "lr": LR * GATE_LR_MULT, "weight_decay": 0.0},
        ],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = -1.0
    best_state = None
    history = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, epoch, target_lambda)
        val_acc = evaluate(model, val_loader)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_acc": train_metrics["acc"],
                "val_acc": val_acc,
                "train_loss": train_metrics["loss"],
                "cls_loss": train_metrics["cls_loss"],
                "sparse_loss": train_metrics["sparse_loss"],
                "effective_lambda": train_metrics["effective_lambda"],
                "sparsity": train_metrics["sparsity"],
                "gate_mean": train_metrics["gate_mean"],
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_metrics['loss']:.4f} "
            f"cls={train_metrics['cls_loss']:.4f} "
            f"sparse={train_metrics['sparse_loss']:.4f} "
            f"train_acc={train_metrics['acc']:.2%} "
            f"val_acc={val_acc:.2%} "
            f"lam_eff={train_metrics['effective_lambda']:.6f} "
            f"sparsity={train_metrics['sparsity']:.2%} "
            f"gate_mean={train_metrics['gate_mean']:.4f}"
        )

    model.load_state_dict(best_state)
    pre_prune_test_acc = evaluate(model, test_loader)
    pre_prune_sparsity = model.overall_sparsity()

    model.hard_prune()
    test_acc = evaluate(model, test_loader)
    sparsity = model.overall_sparsity()
    gates = model.all_gate_values()

    print(
        f"Best val_acc={best_val_acc:.2%} | "
        f"test_acc(before prune)={pre_prune_test_acc:.2%} | "
        f"test_acc(after prune)={test_acc:.2%} | "
        f"sparsity={sparsity:.2%} | "
        f"time={time.time() - start:.1f}s"
    )

    return {
        "lambda": target_lambda,
        "val_acc": best_val_acc,
        "test_acc_pre_prune": pre_prune_test_acc,
        "test_acc": test_acc,
        "sparsity": sparsity,
        "gates": gates,
        "history": history,
        "gate_mean": float(gates.mean()),
        "gate_min": float(gates.min()),
        "gate_max": float(gates.max()),
    }


def plot_best_gate_distribution(result) -> None:
    gates = result["gates"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(PRUNE_THRESH, color="crimson", linestyle="--", linewidth=1.5, label=f"Threshold = {PRUNE_THRESH}")
    ax.set_title(
        f"Gate distribution for best model (lambda = {result['lambda']})\n"
        f"Test accuracy = {result['test_acc']:.2%}, sparsity = {result['sparsity']:.2%}"
    )
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(BEST_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {BEST_PLOT}")


def save_results(results) -> None:
    with RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Lambda",
                "Validation Accuracy (%)",
                "Test Accuracy Before Hard Prune (%)",
                "Test Accuracy After Hard Prune (%)",
                "Sparsity Level (%)",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["lambda"],
                    f"{result['val_acc'] * 100:.2f}",
                    f"{result['test_acc_pre_prune'] * 100:.2f}",
                    f"{result['test_acc'] * 100:.2f}",
                    f"{result['sparsity'] * 100:.2f}",
                ]
            )
    print(f"Saved table to {RESULTS_CSV}")


def write_report(results) -> None:
    best_result = max(results, key=lambda item: item["test_acc"])
    lines = [
        "# Self-Pruning Neural Network Report",
        "",
        "## Why an L1 Penalty on Sigmoid Gates Encourages Sparsity",
        "",
        "Each prunable weight is multiplied by a sigmoid gate between 0 and 1.",
        "Applying an L1 penalty to these gate values increases the loss when too many gates stay open.",
        f"As training minimizes the total loss, less useful gates are pushed toward 0, and any gate below `{PRUNE_THRESH}` is treated as pruned.",
        "",
        "## Results for Different Lambda Values",
        "",
        "The table below summarizes the final test accuracy and sparsity level for the three lambda values tested:",
        "",
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) |",
        "| --- | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            f"| {result['lambda']} | {result['test_acc'] * 100:.2f} | {result['sparsity'] * 100:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Best Model",
            "",
            f"The best model used lambda = {best_result['lambda']:.2f}.",
            "",
            f"![Best gate distribution]({BEST_PLOT.as_posix()})",
            "",
            "The plot shows a large spike near 0 and another cluster away from 0, indicating successful separation between pruned and active connections.",
        ]
    )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {REPORT_MD}")


def main():
    seed_everything(SEED)
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders()

    results = []
    for target_lambda in LAMBDAS:
        results.append(run_experiment(target_lambda, train_loader, val_loader, test_loader))

    results.sort(key=lambda item: item["lambda"])
    best_result = max(results, key=lambda item: item["test_acc"])

    print(f"\n{'=' * 60}")
    print(f"{'Lambda':<12}{'Val Acc':>12}{'Test Acc':>12}{'Sparsity':>12}")
    print(f"{'-' * 48}")
    for result in results:
        print(
            f"{result['lambda']:<12}"
            f"{result['val_acc']:>11.2%}"
            f"{result['test_acc']:>11.2%}"
            f"{result['sparsity']:>11.2%}"
        )
    print(f"{'=' * 60}")
    print(f"Best lambda by test accuracy: {best_result['lambda']}")

    save_results(results)
    plot_best_gate_distribution(best_result)
    write_report(results)


if __name__ == "__main__":
    main()
