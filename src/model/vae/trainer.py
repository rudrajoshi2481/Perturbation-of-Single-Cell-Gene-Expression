import os
import json
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from vae_model import VAE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm


def build_sm_name_to_idx(pairs):
    """Map unique sm_name strings to integer indices for the Embedding layer."""
    unique_names = sorted({p["sm_name"] for p in pairs})
    return {name: idx for idx, name in enumerate(unique_names)}


class TrainDataset(Dataset):
    def __init__(self, path="/app/cell_line_Database/EndToEnd/src/preprocessing/train_data.pt"):
        super().__init__()
        data = torch.load(path, weights_only=False)
        self.pairs = data
        self.sm_name_to_idx = build_sm_name_to_idx(data)
        self.num_drugs = len(self.sm_name_to_idx)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        control_de = torch.tensor(item["control_de"].to_numpy(dtype="float32"))
        treated_de = torch.tensor(item["treated_de"].to_numpy(dtype="float32"))
        drug_idx = self.sm_name_to_idx[item["sm_name"]]
        return control_de, treated_de, drug_idx


def vae_loss(recon, target, mu, logvar, kl_weight=1.0):
    """VAE ELBO: reconstruction MSE + KL divergence (per-element)."""
    # Per-gene MSE so it's comparable to per-latent-dim KL
    recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl, recon_loss, kl


def compute_metrics(recon, target, mu, logvar):
    """Compute per-batch metrics."""
    with torch.no_grad():
        mse = nn.functional.mse_loss(recon, target, reduction="mean").item()
        mae = nn.functional.l1_loss(recon, target, reduction="mean").item()
        # Pearson correlation per sample, then mean
        x = recon - recon.mean(dim=1, keepdim=True)
        y = target - target.mean(dim=1, keepdim=True)
        num = (x * y).sum(dim=1)
        den = (x.pow(2).sum(dim=1).sqrt() * y.pow(2).sum(dim=1).sqrt()) + 1e-8
        corr = (num / den).mean().item()
        # R2 (explained variance)
        ss_res = ((target - recon) ** 2).sum(dim=1)
        ss_tot = ((target - target.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        r2 = (1 - ss_res / (ss_tot + 1e-8)).mean().item()
        # KL per dimension
        kl_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean().item()
    return mse, mae, corr, r2, kl_dim


def evaluate(model, data_loader, kl_weight, device):
    """Run one evaluation pass and return metrics dict."""
    model.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "mse": 0.0, "mae": 0.0, "corr": 0.0, "r2": 0.0, "kl_dim": 0.0}
    n_batches = 0
    with torch.no_grad():
        for control_de, treated_de, drug_idx in data_loader:
            control_de = control_de.to(device)
            treated_de = treated_de.to(device)
            drug_idx = drug_idx.to(device)
            recon, mu, logvar = model(control_de, drug_idx)
            loss, recon_l, kl_l = vae_loss(recon, treated_de, mu, logvar, kl_weight=kl_weight)
            mse, mae, corr, r2, kl_dim = compute_metrics(recon, treated_de, mu, logvar)
            totals["total"] += loss.item()
            totals["recon"] += recon_l.item()
            totals["kl"] += kl_l.item()
            totals["mse"] += mse
            totals["mae"] += mae
            totals["corr"] += corr
            totals["r2"] += r2
            totals["kl_dim"] += kl_dim
            n_batches += 1
    model.train()
    return {k: v / n_batches for k, v in totals.items()}


def train_model(
    data_path="/app/cell_line_Database/EndToEnd/src/preprocessing/train_data.pt",
    input_dim=18211,
    num_layers=2,
    hidden_dim=128,
    latent_dim=32,
    batch_size=16,
    epochs=50,
    lr=1e-3,
    kl_weight=1.0,
    device="cpu",
    output_dir="/app/cell_line_Database/EndToEnd/src/trash/vae_01",
    val_split=0.15,
    test_split=0.15,
    seed=42,
):
    # ---- Dataset split ----
    full_dataset = TrainDataset(path=data_path)
    n = len(full_dataset)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    os.makedirs(output_dir, exist_ok=True)

    # Setup logger -> console + file
    log_path = os.path.join(output_dir, "training.log")
    logger = logging.getLogger("vae_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logging to: {log_path}")
    logger.info(f"Dataset split | train={n_train} val={n_val} test={n_test} | total={n}")

    # Model
    model = VAE(
        input_dim=input_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    # Verify drug embedding size matches dataset
    if model.drug_emb.num_embeddings < full_dataset.num_drugs:
        logger.warning(f"Expanding drug embedding from {model.drug_emb.num_embeddings} to {full_dataset.num_drugs}")
        model.drug_emb = nn.Embedding(full_dataset.num_drugs, latent_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Metrics tracking (train + val)
    history = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "train_mse": [], "train_mae": [], "train_corr": [], "train_r2": [], "train_kl_dim": [],
        "val_total": [], "val_recon": [], "val_kl": [],
        "val_mse": [], "val_mae": [], "val_corr": [], "val_r2": [], "val_kl_dim": [],
    }
    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "best_model.pt")
    last_path = os.path.join(output_dir, "last_model.pt")

    # ---- Training loop ----
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_mse = 0.0
        epoch_mae = 0.0
        epoch_corr = 0.0
        epoch_r2 = 0.0
        epoch_kl_dim = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for control_de, treated_de, drug_idx in pbar:
            control_de = control_de.to(device)
            treated_de = treated_de.to(device)
            drug_idx = drug_idx.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(control_de, drug_idx)
            loss, recon_l, kl_l = vae_loss(recon, treated_de, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()

            mse, mae, corr, r2, kl_dim = compute_metrics(recon, treated_de, mu, logvar)

            epoch_total += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()
            epoch_mse += mse
            epoch_mae += mae
            epoch_corr += corr
            epoch_r2 += r2
            epoch_kl_dim += kl_dim
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{mse:.4f}",
                "corr": f"{corr:.4f}",
            })

        # Train averages
        avg = {k: v / num_batches for k, v in {
            "total": epoch_total, "recon": epoch_recon, "kl": epoch_kl,
            "mse": epoch_mse, "mae": epoch_mae, "corr": epoch_corr, "r2": epoch_r2, "kl_dim": epoch_kl_dim,
        }.items()}

        # Validation
        val = evaluate(model, val_loader, kl_weight, device)

        for prefix, src in [("train_", avg), ("val_", val)]:
            for k, v in src.items():
                history[f"{prefix}{k}"].append(v)

        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d} | "
            f"TRAIN loss={avg['total']:.4f} recon={avg['recon']:.4f} kl={avg['kl']:.4f} "
            f"mse={avg['mse']:.4f} mae={avg['mae']:.4f} corr={avg['corr']:.4f} r2={avg['r2']:.4f} | "
            f"VAL total={val['total']:.4f} recon={val['recon']:.4f} kl={val['kl']:.4f} "
            f"mse={val['mse']:.4f} corr={val['corr']:.4f}"
        )

        # Save best on validation loss
        if val["total"] < best_val_loss:
            best_val_loss = val["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": best_val_loss,
            }, best_path)
            logger.info(f"Best model saved (val_loss={best_val_loss:.6f}) -> {best_path}")

    # Save last checkpoint
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": avg["total"],
    }, last_path)
    logger.info(f"Last model saved -> {last_path}")

    # ---- Test evaluation ----
    logger.info("Running test evaluation...")
    test = evaluate(model, test_loader, kl_weight, device)
    logger.info(
        f"TEST | total={test['total']:.4f} recon={test['recon']:.4f} kl={test['kl']:.4f} "
        f"mse={test['mse']:.4f} mae={test['mae']:.4f} corr={test['corr']:.4f} r2={test['r2']:.4f}"
    )

    # ---- Publication-quality plots ----
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#222222",
    })

    epochs = np.arange(1, len(history["train_total"]) + 1)
    palette = ["#5B8DB8", "#F4A35A", "#6DBF8A", "#D96B6B", "#A48CC4"]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                           height_ratios=[1, 1], width_ratios=[1, 1, 1])

    # 1. Total Loss (train vs val)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_total"], color=palette[0], linewidth=1.8, label="Train Total")
    ax1.plot(epochs, history["val_total"], color=palette[1], linewidth=1.8, label="Val Total")
    ax1.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax1.set_ylabel("Loss", fontsize=11, labelpad=8)
    ax1.set_title("VAE Total Loss", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    # 2. Recon Loss (train vs val)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["train_recon"], color=palette[0], linewidth=1.8, label="Train Recon")
    ax2.plot(epochs, history["val_recon"], color=palette[1], linewidth=1.8, label="Val Recon")
    ax2.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax2.set_ylabel("Recon Loss", fontsize=11, labelpad=8)
    ax2.set_title("Reconstruction Loss", fontsize=13, fontweight="bold", pad=10)
    ax2.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    # 3. KL Loss (train vs val)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history["train_kl"], color=palette[0], linewidth=1.8, label="Train KL")
    ax3.plot(epochs, history["val_kl"], color=palette[1], linewidth=1.8, label="Val KL")
    ax3.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax3.set_ylabel("KL Loss", fontsize=11, labelpad=8)
    ax3.set_title("KL Divergence", fontsize=13, fontweight="bold", pad=10)
    ax3.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    # 4. MSE & MAE (train)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history["train_mse"], color=palette[0], linewidth=1.8, label="Train MSE")
    ax4.plot(epochs, history["val_mse"], color=palette[1], linewidth=1.8, label="Val MSE")
    ax4.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax4.set_ylabel("MSE", fontsize=11, labelpad=8)
    ax4.set_title("Mean Squared Error", fontsize=13, fontweight="bold", pad=10)
    ax4.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    # 5. Correlation & R2
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history["train_corr"], color=palette[0], linewidth=1.8, label="Train Corr")
    ax5.plot(epochs, history["val_corr"], color=palette[1], linewidth=1.8, label="Val Corr")
    ax5.plot(epochs, history["train_r2"], color=palette[2], linewidth=1.8, label="Train R²")
    ax5.plot(epochs, history["val_r2"], color=palette[3], linewidth=1.8, label="Val R²")
    ax5.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax5.set_ylabel("Score", fontsize=11, labelpad=8)
    ax5.set_title("Reconstruction Quality", fontsize=13, fontweight="bold", pad=10)
    ax5.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    ax5.set_ylim(-1.05, 1.05)

    # 6. KL per dimension
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, history["train_kl_dim"], color=palette[0], linewidth=1.8, label="Train")
    ax6.plot(epochs, history["val_kl_dim"], color=palette[1], linewidth=1.8, label="Val")
    ax6.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax6.set_ylabel("KL (per dim)", fontsize=11, labelpad=8)
    ax6.set_title("KL per Latent Dimension", fontsize=13, fontweight="bold", pad=10)
    ax6.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    fig.suptitle("VAE Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plot_path = os.path.join(output_dir, "training_dashboard.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info(f"Dashboard saved to {plot_path}")

    # Save history JSON
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_path}")

    return model, history


if __name__ == "__main__":
    model, history = train_model()
