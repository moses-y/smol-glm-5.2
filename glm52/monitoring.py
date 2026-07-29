from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "smol-glm52-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TrainingMonitor:
    """Durable JSONL, state, and plot artifacts consumed by the dashboard."""

    def __init__(
        self,
        run_dir: Path,
        model_parameters: int,
        active_parameters: int,
        total_steps: int,
        architecture: dict,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"
        self.state_path = run_dir / "dashboard_state.json"
        self.plot_path = run_dir / "loss.png"
        self.train_points: list[tuple[int, float]] = []
        self.main_points: list[tuple[int, float]] = []
        self.validation_points: list[tuple[int, float]] = []
        self.state = {
            "status": "running",
            "updated_at": _now(),
            "model_parameters": model_parameters,
            "active_parameters": active_parameters,
            "total_steps": total_steps,
            "architecture": architecture,
            "step": None,
            "train_loss": None,
            "main_loss": None,
            "mtp_loss": None,
            "indexer_loss": None,
            "validation_loss": None,
            "tokens_per_second": None,
            "tokens_seen": 0,
            "vram_gib": None,
            "lr_scale": None,
            "grad_norm": None,
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "sample_prompt": None,
            "sample_text": None,
            "sample_step": None,
        }
        self._restore_metrics()
        self._write_state()
        self._draw_plot()

    def _restore_metrics(self) -> None:
        if not self.metrics_path.exists():
            return
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "train":
                self.train_points.append((record["step"], record["loss"]))
                if record.get("main_loss") is not None:
                    self.main_points.append(
                        (record["step"], record["main_loss"])
                    )
            elif record.get("kind") == "validation":
                self.validation_points.append((record["step"], record["loss"]))

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    def _write_state(self) -> None:
        self.state["updated_at"] = _now()
        self._atomic_text(
            self.state_path,
            json.dumps(self.state, indent=2) + "\n",
        )

    def _append_metric(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _draw_plot(self) -> None:
        plt.style.use("dark_background")
        figure, axis = plt.subplots(figsize=(10.5, 4.8), dpi=140)
        figure.patch.set_facecolor("#0b1110")
        axis.set_facecolor("#0b1110")
        if self.train_points:
            steps, losses = zip(*self.train_points)
            axis.plot(
                steps,
                losses,
                color="#d6ff5f",
                linewidth=1.5,
                alpha=0.7,
                label="total objective",
            )
        if self.main_points:
            steps, losses = zip(*self.main_points)
            axis.plot(
                steps,
                losses,
                color="#f4f0df",
                linewidth=1.7,
                label="next-token",
            )
        if self.validation_points:
            steps, losses = zip(*self.validation_points)
            axis.plot(
                steps,
                losses,
                color="#55d6be",
                marker="o",
                markersize=3,
                linewidth=1.2,
                label="validation",
            )
        axis.set_xlabel("optimizer step", color="#8f9a93")
        axis.set_ylabel("cross-entropy / objective", color="#8f9a93")
        axis.grid(color="#293330", linewidth=0.6, alpha=0.75)
        axis.tick_params(colors="#8f9a93")
        for spine in axis.spines.values():
            spine.set_color("#293330")
        if self.train_points or self.validation_points:
            axis.legend(frameon=False, ncol=3)
        figure.tight_layout()
        temporary = self.plot_path.with_suffix(".tmp.png")
        figure.savefig(temporary, facecolor=figure.get_facecolor())
        plt.close(figure)
        temporary.replace(self.plot_path)

    def record_train(
        self,
        step: int,
        loss: float,
        main_loss: float,
        mtp_loss: float | None,
        indexer_loss: float | None,
        lr_scale: float,
        tokens_per_second: float,
        tokens_seen: int,
        vram_gib: float,
        grad_norm: float,
        elapsed_seconds: float,
        eta_seconds: float | None,
    ) -> None:
        record = {
            "kind": "train",
            "step": step,
            "loss": loss,
            "main_loss": main_loss,
            "mtp_loss": mtp_loss,
            "indexer_loss": indexer_loss,
            "lr_scale": lr_scale,
            "tokens_per_second": tokens_per_second,
            "tokens_seen": tokens_seen,
            "vram_gib": vram_gib,
            "grad_norm": grad_norm,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "timestamp": _now(),
        }
        self._append_metric(record)
        self.train_points.append((step, loss))
        self.main_points.append((step, main_loss))
        self.state.update(record)
        self.state.pop("kind", None)
        self.state.pop("loss", None)
        self.state["train_loss"] = loss
        self._draw_plot()
        self._write_state()

    def record_validation(self, step: int, loss: float) -> None:
        self._append_metric(
            {
                "kind": "validation",
                "step": step,
                "loss": loss,
                "timestamp": _now(),
            }
        )
        self.validation_points.append((step, loss))
        self.state.update(step=step, validation_loss=loss)
        self._draw_plot()
        self._write_state()

    def record_sample(self, step: int, prompt: str, text: str) -> None:
        self.state.update(
            sample_step=step,
            sample_prompt=prompt,
            sample_text=text,
        )
        self._write_state()

    def set_status(self, status: str, message: str | None = None) -> None:
        self.state["status"] = status
        if message is not None:
            self.state["message"] = message
        self._write_state()
