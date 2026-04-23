from __future__ import annotations

from collections.abc import Sequence

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """Sample one token id from next-token logits with temperature and top-p."""
    # We expect one score per vocabulary item for a single decoding step.
    # Shape should be exactly (vocab_size,).
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1D of shape (vocab_size,), got shape {tuple(logits.shape)}")
    # temperature controls randomness:
    # - smaller -> sharper distribution (more deterministic)
    # - larger -> flatter distribution (more random)
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    # top_p must be a probability cutoff between 0 and 1.
    # top_p=1.0 means "do not truncate".
    if not 0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    # tau -> 0 gives greedy decoding: pick the single largest logit directly.
    # We skip softmax in this branch because dividing by 0 is undefined.
    if temperature == 0:
        return int(torch.argmax(logits).item())

    # Temperature scaling is applied before softmax:
    #   softmax(v / tau)
    # Doing softmax in float32 is more numerically stable.
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits.to(torch.float32), dim=-1)

    if top_p < 1.0:
        # Sort probabilities from largest to smallest so we can find
        # the smallest "high-mass" prefix of tokens.
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # cumulative[k] = sum of first k+1 sorted probabilities.
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Mark entries whose cumulative mass is still <= top_p.
        # This often excludes long low-probability tails.
        keep_mask = cumulative <= top_p
        # Always keep at least one token (the highest-probability one),
        # even if top_p is very small.
        keep_mask[0] = True
        # Include the first token that crosses top_p so the retained set
        # reaches (or slightly exceeds) the target probability mass.
        first_exceed = torch.nonzero(cumulative > top_p, as_tuple=False)
        if first_exceed.numel() > 0:
            keep_mask[first_exceed[0, 0]] = True

        # Zero out probabilities outside the nucleus.
        filtered_probs = sorted_probs * keep_mask
        # Renormalize so the filtered distribution sums to 1.
        filtered_probs = filtered_probs / filtered_probs.sum()
        # Sample index in the sorted space.
        sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)
        # Map sampled sorted position back to original vocabulary id.
        sampled_token = sorted_indices[sampled_sorted_idx]
        return int(sampled_token.item())

    # If top_p == 1.0, sample from the full temperature-scaled distribution.
    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt: Sequence[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> list[int]:
    """Autoregressively decode tokens from a prompt."""
    # Basic argument checks for predictable behavior.
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    # We require at least one prompt token because this function currently
    # does not define special handling for an empty prompt.
    if len(prompt) == 0:
        raise ValueError("prompt must contain at least one token")
    # We need context_length so we can crop long histories to model capacity.
    if not hasattr(model, "context_length"):
        raise ValueError("model must expose a context_length attribute")

    # Decode on the same device as model parameters.
    device = next(model.parameters()).device
    # Start output sequence as the prompt, then append new tokens step by step.
    generated = list(prompt)

    # Generation should run in eval mode (no dropout, etc.), but we restore
    # the previous mode afterwards so callers can continue training safely.
    was_training = model.training
    model.eval()
    try:
        context_length = int(model.context_length)
        for _ in range(max_new_tokens):
            # Keep only the newest context_length tokens so model input
            # never exceeds its configured context window.
            context = generated[-context_length:]
            # Model expects batched input shape: (batch=1, seq_len).
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            # logits shape: (1, seq_len, vocab_size)
            logits = model(x)
            # Next-token distribution is taken from the final position only.
            next_token_logits = logits[0, -1]

            # Sample one token using configured temperature/top-p decoding.
            next_token = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
            )
            # Append sampled token so it becomes part of next step's context.
            generated.append(next_token)

            # Optional early stop when EOS token is generated.
            if eos_token_id is not None and next_token == eos_token_id:
                break
    finally:
        # Restore original train/eval mode to avoid side effects on caller.
        if was_training:
            model.train()

    return generated
