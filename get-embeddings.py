# ----------------------------
# Embeddings
# ----------------------------
def get_embeddings(
    model_name: str,
    model_type: str,
    texts: list,
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
    pooling: str = "mean",
    trust_remote_code: bool = False,
) -> torch.Tensor:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)

    # Model choice matters: decoder-only should use AutoModelForCausalLM to reliably get hidden_states.
    if model_type == "decoder-only":
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(device)
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(device)
    model.eval()

    # Ensure pad token exists (important for decoder-only)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    # Safe max_len (avoid inf)
    tok_max = getattr(tokenizer, "model_max_length", 1024)
    if tok_max is None or tok_max == float("inf"):
        tok_max = 1024

    cfg_max = getattr(model.config, "max_position_embeddings", tok_max)
    if cfg_max is None or cfg_max == float("inf"):
        cfg_max = tok_max

    max_len = int(min(max_length, tok_max, cfg_max))
    max_len = max(8, min(max_len, 4096))  # sane bounds

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Safety check (prevents device-side assert if bad ids appear)
        ids = inputs["input_ids"]
        vocab = int(getattr(model.config, "vocab_size", 0) or 0)
        if vocab > 0 and (ids.min().item() < 0 or ids.max().item() >= vocab):
            raise ValueError(f"input_ids out of range: [{ids.min().item()}, {ids.max().item()}], vocab={vocab}")

        with torch.no_grad():
            if model_type == "encoder-only":
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state  # [B,T,H]

            elif model_type == "decoder-only":
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]  # [B,T,H]

            elif model_type == "encoder-decoder":
                # Use encoder outputs for embeddings
                encoder = model.get_encoder()
                enc_out = encoder(**inputs)
                hidden = enc_out.last_hidden_state  # [B,T,H]
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            batch_emb = pool_hidden(hidden, inputs["attention_mask"], pooling=pooling)
            all_emb.append(batch_emb.detach().cpu())

    if not all_emb:
        return torch.empty((0, int(getattr(model.config, "hidden_size", 0) or 0)))

    return torch.cat(all_emb, dim=0)