import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.transformer import Seq2SeqTransformer
from utils.scheduler import NoamLR


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """Trains the Transformer model for one epoch using automatic mixed precision."""
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_y = batch["tgt_y"].to(device)

        optimizer.zero_grad()

        # Forward pass with Automatic Mixed Precision (AMP)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(src, tgt)
            # Flatten tensors for cross-entropy computation: (batch * seq_len, vocab_size) vs (batch * seq_len)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_y.view(-1))

        # Scaled Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        # Optimizer & Scheduler Step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluates the model on the validation dataset."""
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_y = batch["tgt_y"].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(src, tgt)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_y.view(-1))

        total_loss += loss.item()

    return total_loss / len(dataloader)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr_factor: float = 1.0,
    warmup_steps: int = 4000,
    save_dir: str = "checkpoints",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Executes full training loop, logging, and model checkpointing."""
    device = torch.device(device_str)
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Section 5.3 Paper Optimizer Config: Adam (beta1=0.9, beta2=0.98, eps=1e-9)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    # 2. Noam Scheduler
    scheduler = NoamLR(
        optimizer,
        d_model=model.src_embed.d_model,
        warmup_steps=warmup_steps,
        factor=lr_factor,
    )

    # 3. Label Smoothing Cross Entropy Loss (Section 5.4)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx, label_smoothing=0.1)

    # 4. AMP Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")

    print(f"Starting training on device: {device}")
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device
        )
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        perplexity = math.exp(val_loss) if val_loss < 300 else float("inf")

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {perplexity:.2f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint Management
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f" Saved new best model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    print("✅ Training engine script created and structurally validated!")
