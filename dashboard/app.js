const el = (id) => document.getElementById(id);

const number = (value, digits = 3) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(digits);

const compact = (value) => {
  if (value === null || value === undefined) return "—";
  return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(value);
};

const duration = (value) => {
  if (value === null || value === undefined) return "—";
  const seconds = Math.max(0, Math.round(Number(value)));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m ${seconds % 60}s`;
};

function showPlot(available) {
  el("loss-plot").hidden = !available;
  el("plot-placeholder").hidden = available;
}

async function refresh() {
  try {
    const response = await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`state request returned ${response.status}`);
    const state = await response.json();
    const status = state.status || "waiting";
    el("status").textContent = status;
    el("status-dot").className = `status-dot ${status}`;
    document.body.dataset.status = status;

    const step = state.step;
    const total = state.total_steps;
    el("step").textContent =
      step === null || step === undefined ? "—" : `${Number(step).toLocaleString()} / ${total ?? "?"}`;
    const progress =
      step === null || step === undefined || !total
        ? 0
        : Math.min(100, ((Number(step) + 1) / Number(total)) * 100);
    el("progress-label").textContent = `${progress.toFixed(1)}%`;
    el("progress-bar").style.width = `${progress}%`;

    el("train-loss").textContent = number(state.train_loss);
    el("main-loss").textContent = number(state.main_loss);
    el("val-loss").textContent = number(state.validation_loss);
    el("mtp-loss").textContent = number(state.mtp_loss);
    el("indexer-loss").textContent = number(state.indexer_loss);
    el("grad-norm").textContent = number(state.grad_norm);
    el("lr-scale").textContent = number(state.lr_scale, 4);
    el("throughput").textContent =
      state.tokens_per_second == null ? "—" : Math.round(state.tokens_per_second).toLocaleString();
    el("vram").textContent =
      state.vram_gib == null ? "—" : `${number(state.vram_gib, 2)} GiB`;
    el("tokens-seen").textContent = `${compact(state.tokens_seen || 0)} tokens seen`;
    el("elapsed").textContent = duration(state.elapsed_seconds);
    el("eta").textContent = duration(state.eta_seconds);
    el("updated").textContent = state.updated_at
      ? `updated ${new Date(state.updated_at).toLocaleTimeString()}`
      : "waiting for metrics";
    el("message").textContent = state.message || "";

    if (state.model_parameters) {
      el("parameter-count").textContent =
        `${(state.model_parameters / 1e6).toFixed(2)}M total · ` +
        `${(state.active_parameters / 1e6).toFixed(2)}M active`;
    }
    if (state.architecture) {
      const a = state.architecture;
      el("architecture").textContent =
        `${a.layers} layers · ${a.full_indexers} full + ${a.shared_indexers} shared indexers · ` +
        `${a.experts_per_token}-of-${a.experts} MoE · MTP ×${a.mtp_depth}`;
    }

    el("sample-step").textContent =
      state.sample_step == null ? "not generated" : `step ${state.sample_step}`;
    el("sample-prompt").textContent =
      state.sample_prompt ? `PROMPT / ${state.sample_prompt}` : "";
    if (state.sample_text) el("sample").textContent = state.sample_text;

    if (step !== null && step !== undefined) {
      const plot = el("loss-plot");
      plot.onload = () => showPlot(true);
      plot.onerror = () => showPlot(false);
      plot.src = `/artifacts/loss.png?t=${Date.now()}`;
    } else {
      showPlot(false);
    }
  } catch (error) {
    el("status").textContent = "disconnected";
    el("status-dot").className = "status-dot error";
    el("updated").textContent = "connection lost";
    el("message").textContent = error.message;
    document.body.dataset.status = "error";
  }
}

refresh();
setInterval(refresh, 2000);
