"""
Autoresearch single-cell MLP training script.

Usage:
    uv run train.py
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# Keep Matplotlib config/cache inside the repo so sandboxed runs do not
# depend on a writable home-directory config path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MPLCONFIGDIR = os.path.join(PROJECT_ROOT, ".cache", "matplotlib")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Project-local paths
# ---------------------------------------------------------------------------

CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "data_cache", "dataset.pt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ---------------------------------------------------------------------------
# Hyperparameters (edit directly for experiments)
# ---------------------------------------------------------------------------

SEED = 42
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_EVAL = 256
HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT = 0.25
NUM_EPOCHS = 300
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MIN_DELTA = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
ACTIVATION = "relu"
LABEL_SMOOTHING = 0.05


def resolve_device(device_name, cuda_id):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cuda_id}")
        return torch.device("cpu")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_logger(run_dir):
    logger = logging.getLogger("experiment")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(os.path.join(run_dir, "log.txt"))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_runtime_diagnostics(logger, requested_device, cuda_id):
    logger.info("Python executable: %s", sys.executable)
    logger.info("Conda env: %s", os.environ.get("CONDA_DEFAULT_ENV", "<unset>"))
    logger.info("Requested device: %s (cuda_id=%s)", requested_device, cuda_id)
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    logger.info("NVIDIA_VISIBLE_DEVICES=%s", os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>"))
    logger.info("Torch version: %s", torch.__version__)
    logger.info("Torch CUDA build: %s", torch.version.cuda)
    logger.info("torch.cuda.is_available()=%s", torch.cuda.is_available())
    logger.info("torch.cuda.device_count()=%s", torch.cuda.device_count())
    logger.info("/dev/nvidia0 exists: %s", os.path.exists("/dev/nvidia0"))
    logger.info("/dev/nvidiactl exists: %s", os.path.exists("/dev/nvidiactl"))

    nvidia_smi_path = shutil.which("nvidia-smi")
    logger.info("nvidia-smi path: %s", nvidia_smi_path or "<not found>")
    if nvidia_smi_path:
        try:
            result = subprocess.run(
                [nvidia_smi_path, "-L"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            stdout = result.stdout.strip() or "<empty>"
            stderr = result.stderr.strip() or "<empty>"
            logger.info("nvidia-smi -L exit_code=%s stdout=%s stderr=%s", result.returncode, stdout, stderr)
        except Exception as exc:
            logger.info("nvidia-smi probe failed: %s: %s", type(exc).__name__, exc)


def build_loader(features, labels, batch_size, shuffle, pin_memory):
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def build_activation(name):
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    return nn.Tanh()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout, activation):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(build_activation(activation))
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        return self.classifier(hidden)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (batch_predictions == labels).sum().item()
            total_examples += labels.size(0)

            predictions.append(batch_predictions.cpu().numpy())
            targets.append(labels.cpu().numpy())

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    return average_loss, accuracy, y_pred, y_true


def save_history_plot(run_dir, train_loss_history, val_loss_history, train_acc_history, val_acc_history):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="train_loss")
    plt.plot(val_loss_history, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="train_acc")
    plt.plot(val_acc_history, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "loss_acc.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Train an MLP on cached single-cell features")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size_train", type=int, default=BATCH_SIZE_TRAIN)
    parser.add_argument("--batch_size_eval", type=int, default=BATCH_SIZE_EVAL)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--min_delta", type=float, default=MIN_DELTA)
    parser.add_argument("--scheduler_factor", type=float, default=SCHEDULER_FACTOR)
    parser.add_argument("--scheduler_patience", type=int, default=SCHEDULER_PATIENCE)
    parser.add_argument("--activation", choices=["tanh", "relu", "gelu", "silu"], default=ACTIVATION)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--cache_path", type=str, default=CACHE_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--cuda_id", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.cache_path):
        raise FileNotFoundError(
            f"Dataset cache not found at {args.cache_path}. Run `uv run prepare.py` first."
        )

    os.makedirs(args.results_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_id}_MLP{args.num_layers}_hidden{args.hidden_dim}_lr{args.lr}_bs{args.batch_size_train}"
    run_dir = os.path.join(args.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger = build_logger(run_dir)

    overall_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_runtime_diagnostics(logger, args.device, args.cuda_id)
    device = resolve_device(args.device, args.cuda_id)
    pin_memory = device.type == "cuda"
    dataset = torch.load(args.cache_path, map_location="cpu", weights_only=False)

    train_features = dataset["train_features"]
    train_labels = dataset["train_labels"]
    val_features = dataset["val_features"]
    val_labels = dataset["val_labels"]
    test_features = dataset["test_features"]
    test_labels = dataset["test_labels"]
    test_original_features = dataset["test_original_features"]
    test_original_labels = dataset["test_original_labels"]
    class_names = dataset["class_names"]

    train_loader = build_loader(train_features, train_labels, args.batch_size_train, shuffle=True, pin_memory=pin_memory)
    val_loader = build_loader(val_features, val_labels, args.batch_size_eval, shuffle=False, pin_memory=pin_memory)
    test_loader = build_loader(test_features, test_labels, args.batch_size_eval, shuffle=False, pin_memory=pin_memory)
    test_original_loader = build_loader(
        test_original_features, test_original_labels, args.batch_size_eval, shuffle=False, pin_memory=pin_memory
    )

    input_dim = train_features.shape[1]
    num_classes = len(class_names)
    model = MLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    logger.info("Using device: %s", device)
    logger.info("Run dir: %s", run_dir)
    logger.info("Cache path: %s", args.cache_path)
    logger.info("Model architecture:\n%s", model)
    logger.info(
        "Configs: seed=%s batchSizeTrain=%s batchSizeEval=%s hiddenDim=%s numLayers=%s dropout=%.4f "
        "numEpochs=%s lr=%.6f weightDecay=%.6f patience=%s minDelta=%.6f schedulerFactor=%.3f "
        "schedulerPatience=%s activation=%s labelSmoothing=%.4f",
        args.seed,
        args.batch_size_train,
        args.batch_size_eval,
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        args.num_epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        args.min_delta,
        args.scheduler_factor,
        args.scheduler_patience,
        args.activation,
        args.label_smoothing,
    )
    logger.info("Model details: totalParams=%s trainableParams=%s", total_params, trainable_params)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    train_start = time.time()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss_avg, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_loss_history.append(train_loss_avg)
        val_loss_history.append(val_loss_avg)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        scheduler.step(val_loss_avg)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %03d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | lr=%.6f",
            epoch,
            train_loss_avg,
            train_acc,
            val_loss_avg,
            val_acc,
            current_lr,
        )

        if val_loss_avg < best_val_loss - args.min_delta:
            best_val_loss = val_loss_avg
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %03d. Best val_loss=%.4f", epoch, best_val_loss)
                break

    training_seconds = time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    sampled_test_loss, sampled_test_acc, sampled_test_pred, sampled_test_true = evaluate(
        model, test_loader, criterion, device
    )
    original_test_loss, original_test_acc, original_test_pred, original_test_true = evaluate(
        model, test_original_loader, criterion, device
    )

    sampled_test_report = classification_report(
        sampled_test_true, sampled_test_pred, target_names=class_names, zero_division=0
    )
    original_test_report = classification_report(
        original_test_true, original_test_pred, target_names=class_names, zero_division=0
    )
    logger.info("Validation accuracy at selected checkpoint: %.6f", final_val_acc)
    logger.info("Sampled test classification report:\n%s", sampled_test_report)
    logger.info("Sampled test accuracy: %.6f", accuracy_score(sampled_test_true, sampled_test_pred))
    logger.info("Original test classification report:\n%s", original_test_report)
    logger.info("Original test accuracy: %.6f", accuracy_score(original_test_true, original_test_pred))

    plot_path = save_history_plot(
        run_dir=run_dir,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        train_acc_history=train_acc_history,
        val_acc_history=val_acc_history,
    )
    logger.info("Saved plot: %s", plot_path)

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device=device) / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    total_seconds = time.time() - overall_start

    print("---")
    print(f"val_acc:           {final_val_acc:.6f}")
    print(f"val_loss:          {final_val_loss:.6f}")
    print(f"test_acc_sampled:  {sampled_test_acc:.6f}")
    print(f"test_loss_sampled: {sampled_test_loss:.6f}")
    print(f"test_acc_original: {original_test_acc:.6f}")
    print(f"test_loss_original: {original_test_loss:.6f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"num_params:        {total_params}")
    print(f"best_epoch:        {best_epoch}")
    print(f"best_val_loss:     {best_val_loss:.6f}")
    print(f"best_val_acc:      {best_val_acc:.6f}")


if __name__ == "__main__":
    main()
