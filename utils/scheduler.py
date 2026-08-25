import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Noam Learning Rate Scheduler as defined in Section 5.3 of
    'Attention Is All You Need' (Vaswani et al., 2017).

    Increases learning rate linearly for warmup_steps, then decays proportionally
    to the inverse square root of the current step number.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1,
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """
        Calculates the learning rate for the current step count.
        """
        # PyTorch scheduler internal step counter
        step = max(1, self._step_count)

        # Paper formula calculation
        scale = (
            self.factor
            * (self.d_model**-0.5)
            * min(step**-0.5, step * (self.warmup_steps**-1.5))
        )

        return [scale for _ in self.optimizer.param_groups]


if __name__ == "__main__":
    # Test Setup: Verify learning rate behavior over 10,000 steps
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamLR(optimizer, d_model=512, warmup_steps=4000, factor=1.0)

    lrs = []
    for _ in range(10000):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    step_1_lr = lrs[0]
    peak_lr = lrs[3999]  # Step 4000 (Peak of warmup phase)
    decay_lr = lrs[9999]  # Step 10000 (Decay phase)

    print(f"Step 1 LR:                   {step_1_lr:.8f}")
    print(f"Step 4000 (Peak Warmup) LR:  {peak_lr:.8f}")
    print(f"Step 10000 (Decay Step) LR:  {decay_lr:.8f}")

    assert peak_lr > step_1_lr, "LR should increase during the linear warmup phase."
    assert peak_lr > decay_lr, "LR should decay following warmup steps."
    print("✅ NoamLR Scheduler passed verification checks!")
