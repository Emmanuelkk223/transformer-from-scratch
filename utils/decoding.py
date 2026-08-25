import torch
import torch.nn.functional as F
from models.transformer import Seq2SeqTransformer


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
) -> torch.Tensor:
    """
    Greedy decoding algorithm: generates tokens step-by-step picking highest probability token.
    """
    model.eval()
    device = src.device

    # Encode source sequence once
    memory = model.encode(src, src_mask)

    # Initialize decoder input with SOS token: shape (batch_size=1, 1)
    ys = torch.ones(1, 1).fill_(sos_idx).type(torch.long).to(device)

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        out, _ = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])  # Only project last token output

        _, next_word = torch.max(logits, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type(torch.long).to(device).fill_(next_word)], dim=1
        )

        if next_word == eos_idx:
            break

    return ys.squeeze(0)


@torch.no_grad()
def beam_search_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    beam_size: int = 5,
    alpha: float = 0.6,
) -> torch.Tensor:
    """
    Beam Search decoding algorithm with length normalization penalty.
    """
    model.eval()
    device = src.device

    # Encode source sequence once
    memory = model.encode(src, src_mask)

    # Beams structure: list of tuples (sequence_tensor, cumulative_log_prob)
    beams = [(torch.ones(1, 1).fill_(sos_idx).type(torch.long).to(device), 0.0)]
    completed_beams = []

    for step in range(max_len - 1):
        candidates = []

        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                completed_beams.append((seq, score))
                continue

            tgt_mask = model.make_tgt_mask(seq)
            out, _ = model.decode(seq, memory, src_mask, tgt_mask)
            log_probs = F.log_softmax(model.generator(out[:, -1]), dim=-1)

            top_log_probs, top_indices = log_probs.topk(beam_size)

            for i in range(beam_size):
                next_word = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                next_score = score + top_log_probs[0, i].item()
                candidates.append((torch.cat([seq, next_word], dim=1), next_score))

        if not candidates:
            break

        # Length normalization penalty
        def len_penalty(s, seq_len):
            return s / (((5 + seq_len) ** alpha) / (6**alpha))

        # Select top-k beams for next step
        candidates = sorted(
            candidates, key=lambda x: len_penalty(x[1], x[0].size(1)), reverse=True
        )
        beams = candidates[:beam_size]

    completed_beams.extend(beams)
    # Sort completed hypotheses by score
    best_seq, _ = max(completed_beams, key=lambda x: len_penalty(x[1], x[0].size(1)))
    return best_seq.squeeze(0)
