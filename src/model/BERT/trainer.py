import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from transformer import Transformer


def build_sm_name_to_idx(pairs):
    unique_names = sorted({p["sm_name"] for p in pairs})
    return {name: idx for idx, name in enumerate(unique_names)}


class GeneDataset(Dataset):
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for control_de, treated_de, drug_idx in tqdm(loader, desc="Train", leave=False):
        control_de = control_de.to(device)
        treated_de = treated_de.to(device)
        drug_idx = drug_idx.to(device)

        optimizer.zero_grad()
        pred = model(control_de, drug_idx)
        loss = criterion(pred, treated_de)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches else 0.0


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for control_de, treated_de, drug_idx in loader:
            control_de = control_de.to(device)
            treated_de = treated_de.to(device)
            drug_idx = drug_idx.to(device)
            pred = model(control_de, drug_idx)
            loss = criterion(pred, treated_de)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches if n_batches else 0.0


def main(
    data_path="/app/cell_line_Database/EndToEnd/src/preprocessing/train_data.pt",
    epochs=1,
    batch_size=1,
    lr=1e-4,
    device="cpu",
    output_dir="/app/cell_line_Database/EndToEnd/src/trash/bert_01",
    val_split=0.15,
    test_split=0.15,
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    # Logger
    log_path = os.path.join(output_dir, "training.log")
    logger = logging.getLogger("bert_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Output directory: {output_dir}")

    # Dataset split
    full_dataset = GeneDataset(path=data_path)
    n = len(full_dataset)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

    logger.info(f"Dataset split | train={n_train} val={n_val} test={n_test} | total={n}")

    # Model
    model = Transformer(
        attn_layers=2,
        embeding_dim=512,
        num_drugs=full_dataset.num_drugs,
        seq_len=18211,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Starting training for {epochs} epoch(s)...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    test_loss = evaluate(model, test_loader, criterion, device)
    logger.info(f"TEST | loss={test_loss:.6f}")

    # Save
    ckpt_path = os.path.join(output_dir, "model.pt")
    torch.save({"model_state_dict": model.state_dict(), "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}, ckpt_path)
    logger.info(f"Model saved to {ckpt_path}")

    return model


if __name__ == "__main__":
    model = main()
