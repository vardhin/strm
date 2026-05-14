"""
Training for the NSRR TRM model.

Uses deep supervision: each training step runs N_sup supervision steps,
where the carry (y, z) is detached between steps. This lets the model
iteratively refine its answer across multiple passes, emulating very
deep networks without backpropping through all of them.

A target is a candidate dict (same shape as search returns):
    {
        "primary_id":   int,
        "secondary_id": int | None,
        "tertiary_id":  int | None,
        "comp_type":    "none" | "sequential" | "nested" | "parallel" | "loop_direct",
    }
"""

import torch
import torch.nn.functional as F

from model import TRM, fresh_carry
from search import format_examples


COMP_TYPE_INDEX = {"none": 0, "sequential": 1, "nested": 2, "parallel": 3, "loop_direct": 0, "loop_binary": 0}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(outputs: dict[str, torch.Tensor], target: dict,
                 batch_size: int) -> torch.Tensor:
    """Multi-head cross-entropy loss for a single target composition."""

    primary_target = torch.full((batch_size,), target["primary_id"], dtype=torch.long)
    secondary_val = target.get("secondary_id") or 0
    secondary_target = torch.full((batch_size,), secondary_val, dtype=torch.long)
    tertiary_val = target.get("tertiary_id") or 0
    tertiary_target = torch.full((batch_size,), tertiary_val, dtype=torch.long)
    comp_target = torch.full((batch_size,), COMP_TYPE_INDEX[target["comp_type"]], dtype=torch.long)

    loss_p = F.cross_entropy(outputs["primary_logits"], primary_target)
    loss_s = F.cross_entropy(outputs["secondary_logits"], secondary_target)
    loss_t = F.cross_entropy(outputs["tertiary_logits"], tertiary_target)
    loss_c = F.cross_entropy(outputs["composition_logits"], comp_target)
    loss_h = F.binary_cross_entropy_with_logits(
        outputs["halt_logits"], torch.ones(batch_size)
    )

    # Down-weight tertiary and halt when they don't matter
    t_weight = 1.0 if target.get("tertiary_id") is not None else 0.1
    s_weight = 1.0 if target.get("secondary_id") is not None else 0.1

    total = loss_p + s_weight * loss_s + t_weight * loss_t + loss_c + 0.1 * loss_h

    # Entropy regularization: prevent softmax from collapsing to a single function.
    entropy_weight = 0.05
    for key in ("primary_logits", "secondary_logits", "tertiary_logits"):
        probs = F.softmax(outputs[key], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        total = total - entropy_weight * entropy

    return total


# ---------------------------------------------------------------------------
# Training loop with deep supervision
# ---------------------------------------------------------------------------

def train_on_examples(model: TRM, optimizer: torch.optim.Optimizer,
                      examples: list, target: dict, *,
                      input_dim: int, seq_len: int,
                      num_epochs: int = 30, n_sup: int = 16) -> list[float]:
    """Train the model to predict `target` given `examples`.

    Uses deep supervision: each epoch runs n_sup supervision steps,
    accumulating loss at each step. The carry (y, z) persists across
    supervision steps (detached) so the model learns to iteratively
    refine its answer.

    Returns list of loss values per epoch.
    """
    x_input = format_examples(examples, input_dim=input_dim, seq_len=seq_len)
    batch_size = x_input.shape[0]
    losses = []

    model.train()
    for epoch in range(num_epochs):
        # Fresh carry at the start of each epoch
        carry = fresh_carry(batch_size, seq_len, model.d_model)
        epoch_loss = 0.0

        for sup_step in range(n_sup):
            # Forward: one deep supervision step
            # (internally does T*(n+1) transformer applications)
            carry, outputs = model(carry, x_input)

            loss = compute_loss(outputs, target, batch_size)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # ACT: check if model wants to halt (simple version —
            # halt if prediction is confident enough)
            with torch.no_grad():
                halt_prob = torch.sigmoid(outputs["halt_logits"]).mean()
                if halt_prob > 0.5 and sup_step >= 1:
                    break

        avg_loss = epoch_loss / (sup_step + 1)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"    epoch {epoch}: loss = {avg_loss:.4f} ({sup_step + 1} sup steps)")

    return losses


def train_on_replay(model: TRM, optimizer: torch.optim.Optimizer,
                    replay_buffer: list[dict], *,
                    input_dim: int, seq_len: int,
                    epochs_per_task: int = 2, n_sup: int = 16) -> list[float]:
    """Full replay training: shuffle all tasks each epoch, train on everything.

    Each entry in replay_buffer is:
        {"examples": [...], "target": {...}}

    Epochs = len(replay_buffer) * epochs_per_task.
    Each epoch shuffles the buffer and does one pass through all tasks.

    Returns list of average loss per epoch.
    """
    n_tasks = len(replay_buffer)
    if n_tasks == 0:
        return []

    num_epochs = n_tasks * epochs_per_task

    # Pre-encode all tasks
    encoded = []
    for entry in replay_buffer:
        examples = entry["examples"]
        target = entry["target"]
        x_input = format_examples(examples, input_dim=input_dim, seq_len=seq_len)
        batch_size = x_input.shape[0]
        encoded.append({
            "x_input": x_input,
            "target": target,
            "batch_size": batch_size,
        })

    losses = []
    model.train()
    best_loss = float("inf")
    patience = 5
    no_improve = 0

    for epoch in range(num_epochs):
        # Sequential order: learning builds on prior knowledge (curriculum first)
        order = list(range(n_tasks))

        epoch_loss = 0.0
        task_count = 0

        for idx in order:
            enc = encoded[idx]
            x_input = enc["x_input"]
            target = enc["target"]
            batch_size = enc["batch_size"]

            # Fresh carry per task
            carry = fresh_carry(batch_size, seq_len, model.d_model)

            for sup_step in range(n_sup):
                carry, outputs = model(carry, x_input)
                loss = compute_loss(outputs, target, batch_size)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # ACT halt
                with torch.no_grad():
                    halt_prob = torch.sigmoid(outputs["halt_logits"]).mean()
                    if halt_prob > 0.5 and sup_step >= 1:
                        break

            epoch_loss += loss.item()
            task_count += 1

        avg_loss = epoch_loss / max(task_count, 1)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"    replay epoch {epoch}/{num_epochs}: avg_loss = {avg_loss:.4f} ({n_tasks} tasks)")

        # Early stopping: if no improvement for `patience` epochs, stop
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop at epoch {epoch} (no improvement for {patience} epochs, loss={avg_loss:.4f})")
                break

    return losses
