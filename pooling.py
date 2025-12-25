# ----------------------------
# Pooling
# ----------------------------
def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    """
    hidden: [B, T, H]
    attention_mask: [B, T] (1 real token, 0 pad)
    """
    mask = attention_mask.unsqueeze(-1).float()  # [B,T,1]

    if pooling == "mean":
        den = mask.sum(dim=1).clamp(min=1e-9)
        return (hidden * mask).sum(dim=1) / den

    if pooling == "max":
        hidden_masked = hidden.masked_fill(mask == 0, -1e9)
        return hidden_masked.max(dim=1).values

    if pooling == "cls":
        return hidden[:, 0, :]

    if pooling == "last":
        idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
        idx = idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))  # [B,1,H]
        return hidden.gather(dim=1, index=idx).squeeze(1)

    raise ValueError(f"Unknown pooling: {pooling}")
