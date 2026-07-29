from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from glm52 import GLM52Config, GLM52ForCausalLM
from glm52.monitoring import TrainingMonitor
from glm52.optim import GLM52Optimizers


def get_batch(
    data: np.memmap,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= sequence_length + 1:
        raise ValueError("dataset shard is shorter than sequence_length + 1")
    positions = torch.randint(
        0,
        len(data) - sequence_length - 1,
        (batch_size,),
    )
    sequences = [
        torch.from_numpy(
            np.asarray(
                data[position : position + sequence_length + 1],
                dtype=np.int64,
            )
        )
        for position in positions.tolist()
    ]
    batch = torch.stack(sequences)
    return (
        batch[:, :-1].to(device, non_blocking=True),
        batch[:, 1:].to(device, non_blocking=True),
    )


def learning_rate_scale(
    step: int,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> float:
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This 50M run is intended for a local GPU. "
            "Use --device cpu only for a very small smoke configuration."
        )
    return torch.device("cuda")


def autocast_factory(device: torch.device):
    if device.type == "cuda":
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext


@torch.no_grad()
def evaluate(
    model: GLM52ForCausalLM,
    data: np.memmap,
    batches: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    autocast_context,
) -> float:
    model.eval()
    losses: list[torch.Tensor] = []
    for _ in range(batches):
        input_ids, labels = get_batch(
            data,
            batch_size,
            sequence_length,
            device,
        )
        with autocast_context():
            output = model(input_ids, labels, compute_auxiliary=False)
        if output.main_loss is None:
            raise RuntimeError("evaluation did not produce a next-token loss")
        losses.append(output.main_loss.detach().float())
    model.train()
    return torch.stack(losses).mean().item()


def checkpoint_payload(
    model: GLM52ForCausalLM,
    optimizers: GLM52Optimizers,
    step: int,
    best_validation_loss: float,
    train_args: dict,
) -> dict:
    payload = {
        "model": model.state_dict(),
        "optimizers": optimizers.state_dict(),
        "config": model.config.to_dict(),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "train_args": train_args,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(
    path: Path,
    model: GLM52ForCausalLM,
    optimizers: GLM52Optimizers,
    step: int,
    best_validation_loss: float,
    train_args: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        checkpoint_payload(
            model,
            optimizers,
            step,
            best_validation_loss,
            train_args,
        ),
        temporary,
    )
    temporary.replace(path)


def restore_rng(checkpoint: dict) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])


@torch.inference_mode()
def generate_sample(
    model: GLM52ForCausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    autocast_context,
) -> str:
    was_training = model.training
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt).ids],
        dtype=torch.long,
        device=device,
    )
    with autocast_context():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            eos_token_id=tokenizer.token_to_id("<eos>"),
        )
    if was_training:
        model.train()
    return tokenizer.decode(
        output_ids[0].tolist(),
        skip_special_tokens=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train smol-glm-5.2 on the prepared TinyStories corpus"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/glm52_50m.json"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tinystories"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/glm52-50m"),
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--muon-lr", type=float, default=0.01)
    parser.add_argument("--adamw-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=250)
    parser.add_argument("--sample-tokens", type=int, default=64)
    parser.add_argument(
        "--sample-prompt",
        default="Once upon a time, a little fox",
    )
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0 or args.gradient_accumulation <= 0:
        parser.error("batch size and gradient accumulation must be positive")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    config = GLM52Config.from_json(args.config)
    metadata_path = args.data_dir / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["vocab_size"] != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab {metadata['vocab_size']} does not match "
            f"model vocab {config.vocab_size}"
        )
    if metadata.get("dtype") != "uint16":
        raise ValueError("prepared data metadata must declare uint16 tokens")
    train_data = np.memmap(
        args.data_dir / "train.bin",
        dtype=np.uint16,
        mode="r",
    )
    validation_data = np.memmap(
        args.data_dir / "validation.bin",
        dtype=np.uint16,
        mode="r",
    )

    model = GLM52ForCausalLM(config).to(device)
    model_parameters = model.num_parameters()
    active_parameters = model.num_active_parameters()
    print(
        f"parameters: {model_parameters:,} total / "
        f"{active_parameters:,} active per token"
    )
    print(
        f"architecture: {config.num_layers} layers, "
        f"{sum(kind == 'full' for kind in config.indexer_types)} full indexers, "
        f"{sum(kind == 'shared' for kind in config.indexer_types)} shared indexers"
    )

    optimizers = GLM52Optimizers(
        model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    best_validation_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizers.load_state_dict(checkpoint["optimizers"])
        start_step = checkpoint["step"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        restore_rng(checkpoint)
        print(f"resumed after step {checkpoint['step']}")
    if start_step >= args.steps:
        raise ValueError(
            f"resume checkpoint is already at step {start_step - 1}; "
            f"--steps must be greater than {start_step}"
        )

    autocast_context = autocast_factory(device)
    tokenizer = Tokenizer.from_file(
        str(args.data_dir / "tokenizer.json")
    )
    monitor = TrainingMonitor(
        args.output_dir,
        model_parameters=model_parameters,
        active_parameters=active_parameters,
        total_steps=args.steps,
        architecture={
            "name": "smol-glm-5.2",
            "layers": config.num_layers,
            "full_indexers": sum(
                kind == "full" for kind in config.indexer_types
            ),
            "shared_indexers": sum(
                kind == "shared" for kind in config.indexer_types
            ),
            "dense_layers": config.dense_layers,
            "moe_layers": config.num_layers - config.dense_layers,
            "experts": config.num_routed_experts,
            "experts_per_token": config.experts_per_token,
            "mtp_depth": config.mtp_depth,
        },
    )
    train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    model.train()
    optimizers.zero_grad()
    run_started = time.perf_counter()
    log_started = run_started
    tokens_since_log = 0
    tokens_per_step = (
        args.batch_size
        * args.sequence_length
        * args.gradient_accumulation
    )
    tokens_seen = start_step * tokens_per_step
    last_completed_step = start_step - 1

    def request_stop(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_stop)
    try:
        for step in range(start_step, args.steps):
            scale = learning_rate_scale(
                step,
                args.steps,
                args.warmup_fraction,
                args.minimum_lr_ratio,
            )
            optimizers.set_lr_scale(scale)
            total_loss = 0.0
            main_loss = 0.0
            mtp_loss = 0.0
            indexer_loss = 0.0
            mtp_count = 0
            indexer_count = 0

            for _ in range(args.gradient_accumulation):
                input_ids, labels = get_batch(
                    train_data,
                    args.batch_size,
                    args.sequence_length,
                    device,
                )
                with autocast_context():
                    output = model(
                        input_ids,
                        labels,
                        compute_auxiliary=True,
                    )
                    if output.loss is None or output.main_loss is None:
                        raise RuntimeError("training output is missing losses")
                    scaled_loss = output.loss / args.gradient_accumulation
                scaled_loss.backward()
                total_loss += output.loss.detach().float().item()
                main_loss += output.main_loss.detach().float().item()
                if output.mtp_loss is not None:
                    mtp_loss += output.mtp_loss.detach().float().item()
                    mtp_count += 1
                if output.indexer_loss is not None:
                    indexer_loss += output.indexer_loss.detach().float().item()
                    indexer_count += 1

            divisor = args.gradient_accumulation
            total_loss /= divisor
            main_loss /= divisor
            mtp_value = mtp_loss / mtp_count if mtp_count else None
            indexer_value = (
                indexer_loss / indexer_count if indexer_count else None
            )
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            grad_norm = float(grad_norm_tensor.detach().float())
            optimizers.step()
            optimizers.zero_grad()
            last_completed_step = step
            tokens_since_log += tokens_per_step
            tokens_seen += tokens_per_step

            if step % args.log_interval == 0:
                now = time.perf_counter()
                log_elapsed = max(now - log_started, 1e-9)
                throughput = tokens_since_log / log_elapsed
                elapsed = now - run_started
                completed_this_run = step - start_step + 1
                seconds_per_step = elapsed / completed_this_run
                eta = seconds_per_step * (args.steps - step - 1)
                vram_gib = (
                    torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda"
                    else 0.0
                )
                print(
                    f"step {step:6d} | objective {total_loss:.4f} | "
                    f"next-token {main_loss:.4f} | "
                    f"mtp {mtp_value if mtp_value is not None else float('nan'):.4f} | "
                    f"index {indexer_value if indexer_value is not None else float('nan'):.4f} | "
                    f"{throughput:,.0f} tok/s | vram {vram_gib:.2f} GiB"
                )
                monitor.record_train(
                    step=step,
                    loss=total_loss,
                    main_loss=main_loss,
                    mtp_loss=mtp_value,
                    indexer_loss=indexer_value,
                    lr_scale=scale,
                    tokens_per_second=throughput,
                    tokens_seen=tokens_seen,
                    vram_gib=vram_gib,
                    grad_norm=grad_norm,
                    elapsed_seconds=elapsed,
                    eta_seconds=eta,
                )
                log_started = now
                tokens_since_log = 0

            if step % args.eval_interval == 0 or step == args.steps - 1:
                validation_loss = evaluate(
                    model,
                    validation_data,
                    args.eval_batches,
                    args.batch_size,
                    args.sequence_length,
                    device,
                    autocast_context,
                )
                print(
                    f"step {step:6d} | validation next-token "
                    f"{validation_loss:.4f}"
                )
                monitor.record_validation(step, validation_loss)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model,
                        optimizers,
                        step,
                        best_validation_loss,
                        train_args,
                    )

            should_sample = (
                not args.skip_samples
                and (
                    step % args.sample_interval == 0
                    or step == args.steps - 1
                )
            )
            if should_sample:
                try:
                    sample = generate_sample(
                        model,
                        tokenizer,
                        args.sample_prompt,
                        args.sample_tokens,
                        device,
                        autocast_context,
                    )
                    monitor.record_sample(
                        step,
                        args.sample_prompt,
                        sample,
                    )
                    print(f"step {step:6d} | sample: {sample[:160]!r}")
                except Exception as error:
                    monitor.set_status(
                        "running",
                        f"Sample generation failed at step {step}: {error}",
                    )
                    print(
                        f"step {step:6d} | sample generation failed: {error}"
                    )

            if step % args.save_interval == 0 or step == args.steps - 1:
                save_checkpoint(
                    args.output_dir / "latest.pt",
                    model,
                    optimizers,
                    step,
                    best_validation_loss,
                    train_args,
                )
        monitor.set_status("complete")
    except KeyboardInterrupt:
        if last_completed_step >= start_step:
            save_checkpoint(
                args.output_dir / "latest.pt",
                model,
                optimizers,
                last_completed_step,
                best_validation_loss,
                train_args,
            )
        monitor.set_status(
            "stopped",
            "Training was stopped by the user; latest.pt was preserved.",
        )
        print("training stopped; latest checkpoint preserved")
    except Exception as error:
        monitor.set_status("error", str(error))
        raise


if __name__ == "__main__":
    main()
