import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network (FFN) from Section 3.3.
    Expands d_model to d_ff (default 2048) and projects back to d_model.
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return self.w_2(self.dropout(self.relu(self.w_1(x))))
