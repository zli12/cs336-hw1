from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.train_lm as train_lm


def _make_args(**overrides: object) -> argparse.Namespace:
    args = argparse.Namespace(
        train_data=Path("train.npy"),
        val_data=None,
        token_dtype="uint16",
        batch_size=2,
        context_length=4,
        vocab_size=16,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10_000.0,
        max_steps=3,
        learning_rate=1e-3,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        max_grad_norm=1.0,
        max_wallclock_sec=None,
        lr_schedule="fixed",
        min_learning_rate=0.0,
        warmup_iters=100,
        cosine_cycle_iters=None,
        device="cpu",
        seed=0,
        log_every=100,
        val_every=2,
        val_batches=1,
        checkpoint_path=None,
        checkpoint_every=2,
        resume_from=None,
        wandb=False,
        wandb_project="cs336-basics",
        wandb_run_name=None,
        metrics_csv=None,
        no_rms_norm=False,
        post_norm=False,
        no_rope=False,
        ffn_type="swiglu",
        use_sdpa=False,
        bf16=False,
        torch_compile=False,
        compile_mode="reduce-overhead",
        qk_norm=False,
        tie_embeddings=False,
        embed_init_std=None,
        logit_soft_cap=None,
        z_loss_weight=0.0,
        optimizer="adamw",
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_ns_steps=5,
        decay_frac=0.2,
        value_embed_layers=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_load_token_array_npy_and_raw_binary(tmp_path: Path) -> None:
    arr = np.arange(10, dtype=np.uint16)

    npy_path = tmp_path / "tokens.npy"
    np.save(npy_path, arr)
    loaded_npy = train_lm.load_token_array(npy_path, token_dtype="uint16")
    np.testing.assert_array_equal(np.asarray(loaded_npy), arr)

    raw_path = tmp_path / "tokens.bin"
    arr.tofile(raw_path)
    loaded_raw = train_lm.load_token_array(raw_path, token_dtype="uint16")
    np.testing.assert_array_equal(np.asarray(loaded_raw), arr)


def test_load_token_array_rejects_non_1d(tmp_path: Path) -> None:
    bad = np.arange(12, dtype=np.uint16).reshape(3, 4)
    npy_path = tmp_path / "bad.npy"
    np.save(npy_path, bad)

    with pytest.raises(ValueError, match="Expected 1D token array"):
        train_lm.load_token_array(npy_path, token_dtype="uint16")


def test_evaluate_averages_losses_and_restores_train_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((*x.shape, 5), dtype=torch.float32, device=x.device)

    model = DummyModel()
    model.train()
    calls: list[int] = []
    losses = iter([1.0, 2.0, 3.0])

    def fake_get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(len(dataset))
        x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        return x, y

    def fake_cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _ = logits, y
        return torch.tensor(next(losses), dtype=torch.float32)

    monkeypatch.setattr(train_lm, "get_batch", fake_get_batch)
    monkeypatch.setattr(train_lm, "cross_entropy", fake_cross_entropy)

    out = train_lm.evaluate(
        model=model,
        data=np.arange(100, dtype=np.uint16),
        batch_size=2,
        context_length=4,
        device="cpu",
        num_batches=3,
    )

    assert out == pytest.approx(2.0)
    assert model.training is True
    assert len(calls) == 3


def test_maybe_init_wandb_handles_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _make_args(wandb=True)
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert train_lm.maybe_init_wandb(args, {}) is None


def test_maybe_init_wandb_initializes_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_init(*, project: str, name: str | None, config: dict[str, object]) -> None:
        captured["project"] = project
        captured["name"] = name
        captured["config"] = config

    fake_wandb = SimpleNamespace(init=fake_init)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    args = _make_args(wandb=True, wandb_project="proj", wandb_run_name="run-1")
    out = train_lm.maybe_init_wandb(args, {"k": "v"})

    assert out is fake_wandb
    assert captured == {"project": "proj", "name": "run-1", "config": {"k": "v"}}


def test_main_runs_val_and_checkpoint_cadence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    val_calls: list[int] = []
    saved_steps: list[int] = []

    args = _make_args(
        val_data=tmp_path / "val.npy",
        max_steps=3,
        val_every=2,
        checkpoint_path=tmp_path / "ckpt.pt",
        checkpoint_every=2,
        max_grad_norm=0.0,
    )

    def fake_load_token_array(path: Path, token_dtype: str) -> np.ndarray:
        _ = token_dtype
        if path == args.train_data:
            return np.arange(50, dtype=np.uint16)
        return np.arange(40, dtype=np.uint16)

    def fake_get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        _ = dataset
        x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        return x, y

    def fake_evaluate(
        model: train_lm.TransformerLM,
        data: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str,
        num_batches: int,
        use_bf16: bool = False,
    ) -> float:
        _ = model, data, batch_size, context_length, device, num_batches, use_bf16
        val_calls.append(1)
        return 0.123

    def fake_save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: Path,
    ) -> None:
        _ = model, optimizer, out
        saved_steps.append(iteration)

    def fail_gradient_clipping(parameters: object, max_norm: float) -> None:
        _ = parameters, max_norm
        raise AssertionError("gradient_clipping should not be called when max_grad_norm <= 0")

    monkeypatch.setattr(train_lm, "parse_args", lambda: args)
    monkeypatch.setattr(train_lm, "load_token_array", fake_load_token_array)
    monkeypatch.setattr(train_lm, "get_batch", fake_get_batch)
    monkeypatch.setattr(train_lm, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_lm, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(train_lm, "gradient_clipping", fail_gradient_clipping)

    train_lm.main()

    assert len(val_calls) == 1
    assert saved_steps == [2, 3]


def test_main_resumes_from_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded: list[Path] = []
    saved_steps: list[int] = []

    resume_path = tmp_path / "resume.pt"
    args = _make_args(
        max_steps=5,
        resume_from=resume_path,
        checkpoint_path=tmp_path / "ckpt.pt",
        checkpoint_every=2,
    )

    def fake_load_token_array(path: Path, token_dtype: str) -> np.ndarray:
        _ = path, token_dtype
        return np.arange(100, dtype=np.uint16)

    def fake_get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        _ = dataset
        x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        return x, y

    def fake_load_checkpoint(src: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
        _ = model, optimizer
        loaded.append(src)
        return 3

    def fake_save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: Path,
    ) -> None:
        _ = model, optimizer, out
        saved_steps.append(iteration)

    monkeypatch.setattr(train_lm, "parse_args", lambda: args)
    monkeypatch.setattr(train_lm, "load_token_array", fake_load_token_array)
    monkeypatch.setattr(train_lm, "get_batch", fake_get_batch)
    monkeypatch.setattr(train_lm, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(train_lm, "save_checkpoint", fake_save_checkpoint)

    train_lm.main()

    assert loaded == [resume_path]
    assert saved_steps == [4, 5]


def test_main_writes_step_and_wallclock_metrics_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    args = _make_args(
        val_data=tmp_path / "val.npy",
        max_steps=2,
        log_every=1,
        val_every=2,
        metrics_csv=metrics_path,
        checkpoint_path=None,
    )

    def fake_load_token_array(path: Path, token_dtype: str) -> np.ndarray:
        _ = token_dtype
        if path == args.train_data:
            return np.arange(60, dtype=np.uint16)
        return np.arange(40, dtype=np.uint16)

    def fake_get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        _ = dataset
        x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        return x, y

    monkeypatch.setattr(train_lm, "parse_args", lambda: args)
    monkeypatch.setattr(train_lm, "load_token_array", fake_load_token_array)
    monkeypatch.setattr(train_lm, "get_batch", fake_get_batch)

    train_lm.main()

    lines = metrics_path.read_text().strip().splitlines()
    assert lines[0] == "step,wallclock_sec,split,loss,tokens_per_sec,lr"
    # Two train logs (steps 1 and 2) and one val log (step 2).
    assert len(lines) == 4
    assert any(",train," in line for line in lines[1:])
    assert any(",val," in line for line in lines[1:])


def test_main_applies_cosine_lr_schedule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cosine schedule should write per-step LR into the CSV: ramp up during warmup then decay."""
    metrics_path = tmp_path / "metrics.csv"
    args = _make_args(
        max_steps=10,
        log_every=1,
        val_every=100,
        metrics_csv=metrics_path,
        learning_rate=1e-2,
        min_learning_rate=1e-4,
        lr_schedule="cosine",
        warmup_iters=3,
        cosine_cycle_iters=10,
        max_grad_norm=0.0,
    )

    def fake_load_token_array(path: Path, token_dtype: str) -> np.ndarray:
        _ = path, token_dtype
        return np.arange(40, dtype=np.uint16)

    def fake_get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        _ = dataset
        x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        return x, y

    monkeypatch.setattr(train_lm, "parse_args", lambda: args)
    monkeypatch.setattr(train_lm, "load_token_array", fake_load_token_array)
    monkeypatch.setattr(train_lm, "get_batch", fake_get_batch)

    train_lm.main()

    # Parse the CSV and pull (step, lr) for train rows.
    import csv

    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        train_rows = [(int(r["step"]), float(r["lr"])) for r in reader if r["split"] == "train"]

    train_rows.sort()
    assert len(train_rows) == 10, f"Expected 10 train rows, got {len(train_rows)}: {train_rows}"

    lrs = [lr for (_, lr) in train_rows]
    # During warmup (steps 1..3), LR should strictly increase from <max to max.
    assert lrs[0] < lrs[1] < lrs[2], f"Warmup not strictly increasing in first 3 steps: {lrs[:3]}"
    # Peak should land at step warmup_iters=3.
    assert lrs[2] == pytest.approx(args.learning_rate, rel=1e-6), f"Peak LR != max: {lrs[2]}"
    # After warmup, cosine decay should monotonically (non-strictly) decrease toward min_lr.
    for i in range(3, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-12, f"Cosine not non-increasing at step {i + 1}: {lrs[i - 1]} -> {lrs[i]}"
    # End of cycle should be at min_learning_rate.
    assert lrs[-1] == pytest.approx(args.min_learning_rate, rel=1e-6), f"Final LR != min: {lrs[-1]}"


def test_main_writes_lr_for_fixed_schedule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fixed schedule should still log a constant LR equal to --learning-rate in every CSV row."""
    metrics_path = tmp_path / "metrics.csv"
    args = _make_args(
        max_steps=3,
        log_every=1,
        val_every=100,
        metrics_csv=metrics_path,
        learning_rate=7.5e-4,
        lr_schedule="fixed",
        max_grad_norm=0.0,
    )

    def fake_load_token_array(path: Path, token_dtype: str) -> np.ndarray:
        _ = path, token_dtype
        return np.arange(40, dtype=np.uint16)

    def fake_get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        _ = dataset
        x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        return x, y

    monkeypatch.setattr(train_lm, "parse_args", lambda: args)
    monkeypatch.setattr(train_lm, "load_token_array", fake_load_token_array)
    monkeypatch.setattr(train_lm, "get_batch", fake_get_batch)

    train_lm.main()

    import csv

    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        lrs = [float(r["lr"]) for r in reader]

    assert lrs and all(lr == pytest.approx(args.learning_rate, rel=1e-9) for lr in lrs), lrs


def test_main_wallclock_guard_triggers_early_val_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When --max-wallclock-sec is exceeded, training must stop early with a
    final val pass and a forced checkpoint write at the wallclock-terminated
    step, even though step % checkpoint_every and step % val_every don't
    coincide with that step.
    """
    val_calls: list[int] = []
    saved_steps: list[int] = []
    metrics_path = tmp_path / "metrics.csv"

    args = _make_args(
        val_data=tmp_path / "val.npy",
        max_steps=100,
        log_every=1,
        val_every=50,
        checkpoint_path=tmp_path / "ckpt.pt",
        checkpoint_every=50,
        metrics_csv=metrics_path,
        max_wallclock_sec=0.5,
        max_grad_norm=0.0,
    )

    def fake_load_token_array(path: Path, token_dtype: str) -> np.ndarray:
        _ = token_dtype
        if path == args.train_data:
            return np.arange(60, dtype=np.uint16)
        return np.arange(40, dtype=np.uint16)

    def fake_get_batch(
        dataset: np.ndarray, batch_size: int, context_length: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = dataset
        x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        return x, y

    def fake_evaluate(
        model: train_lm.TransformerLM,
        data: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str,
        num_batches: int,
        use_bf16: bool = False,
    ) -> float:
        _ = model, data, batch_size, context_length, device, num_batches, use_bf16
        val_calls.append(1)
        return 0.456

    def fake_save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: Path,
    ) -> None:
        _ = model, optimizer, out
        saved_steps.append(iteration)

    # Drive a deterministic wallclock: the first time.time() call sets
    # start_time=0.0; every subsequent call returns +1.0s. The wallclock
    # check inside the loop fires on the very first iteration because
    # 1.0 > max_wallclock_sec=0.5.
    counter = {"n": 0}

    def fake_time() -> float:
        n = counter["n"]
        counter["n"] += 1
        return float(n)

    monkeypatch.setattr(train_lm, "parse_args", lambda: args)
    monkeypatch.setattr(train_lm, "load_token_array", fake_load_token_array)
    monkeypatch.setattr(train_lm, "get_batch", fake_get_batch)
    monkeypatch.setattr(train_lm, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_lm, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(train_lm.time, "time", fake_time)

    train_lm.main()

    # Loop should have exited at step 1 via the wallclock break, NOT run to
    # step 100. We expect exactly one forced val and one forced checkpoint
    # at step=1, plus the post-loop final checkpoint also at step=1.
    assert len(val_calls) == 1, val_calls
    assert saved_steps == [1, 1], saved_steps

    # CSV should contain a val row at step 1 (proving the val pass ran on the
    # wallclock-terminated step rather than waiting for step % val_every == 0).
    import csv as _csv

    with metrics_path.open() as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    val_steps = [int(r["step"]) for r in rows if r["split"] == "val"]
    assert val_steps == [1], val_steps
