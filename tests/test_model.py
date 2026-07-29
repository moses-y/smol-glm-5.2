import json

import torch
from safetensors.torch import save_model

from generate import load_trained_model
from glm52 import GLM52Config, GLM52ForCausalLM
from glm52.monitoring import TrainingMonitor


def tiny_config(index_topk: int = 16) -> GLM52Config:
    return GLM52Config(
        vocab_size=128,
        hidden_size=32,
        num_layers=5,
        num_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=index_topk,
        dense_layers=1,
        dense_intermediate_size=64,
        num_routed_experts=4,
        num_shared_experts=1,
        experts_per_token=2,
        moe_intermediate_size=16,
        routed_scaling_factor=1.0,
        mtp_depth=2,
        mtp_intermediate_size=32,
    )


def test_glm52_layer_schedules_match_public_config() -> None:
    config = GLM52Config(num_layers=13, dense_layers=3)
    assert config.indexer_types == [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
        "shared",
    ]
    assert config.mlp_types == ["dense"] * 3 + ["sparse"] * 10
    model = GLM52ForCausalLM(tiny_config())
    assert model.layers[2].attention.indexer is not None
    assert model.layers[3].attention.indexer is None


def test_50m_configuration_parameter_budget() -> None:
    config = GLM52Config.from_json("configs/glm52_50m.json")
    model = GLM52ForCausalLM(config)
    assert model.num_parameters() == 50_153_344
    assert 20_000_000 < model.num_active_parameters() < model.num_parameters()


def test_forward_backward_auxiliary_losses_and_indexer_gradient() -> None:
    torch.manual_seed(7)
    model = GLM52ForCausalLM(tiny_config(index_topk=4))
    input_ids = torch.randint(0, 128, (2, 8))
    labels = torch.randint(0, 128, (2, 8))
    output = model(input_ids, labels)
    assert output.logits.shape == (2, 8, 128)
    for loss in (
        output.loss,
        output.main_loss,
        output.mtp_loss,
        output.indexer_loss,
    ):
        assert loss is not None and torch.isfinite(loss)
    output.loss.backward()
    indexer = model.layers[0].attention.indexer
    assert indexer is not None
    assert indexer.q_proj.weight.grad is not None
    assert torch.count_nonzero(indexer.q_proj.weight.grad) > 0
    for layer in model.layers[1:]:
        if hasattr(layer.mlp, "router"):
            bias = layer.mlp.router.expert_score_correction_bias
            assert torch.isclose(bias.mean(), torch.tensor(0.0), atol=1e-6)


def test_bfloat16_autocast_forward_backward() -> None:
    """Regression test for dtype-safe causal masks used during GPU training."""
    torch.manual_seed(9)
    model = GLM52ForCausalLM(tiny_config(index_topk=4))
    input_ids = torch.randint(0, 128, (2, 8))
    labels = torch.randint(0, 128, (2, 8))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(input_ids, labels, compute_auxiliary=True)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    indexer = model.layers[0].attention.indexer
    assert indexer is not None
    assert indexer.q_proj.weight.grad is not None
    assert torch.isfinite(indexer.q_proj.weight.grad).all()


def test_model_is_causal_with_sparse_selection() -> None:
    torch.manual_seed(11)
    model = GLM52ForCausalLM(tiny_config(index_topk=3)).eval()
    first = torch.randint(0, 128, (1, 8))
    second = first.clone()
    second[:, 5:] = torch.randint(0, 128, (1, 3))
    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits
    torch.testing.assert_close(
        first_logits[:, :5],
        second_logits[:, :5],
        atol=2e-5,
        rtol=2e-5,
    )


def test_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(13)
    model = GLM52ForCausalLM(tiny_config(index_topk=16)).eval()
    input_ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        full_logits = model(input_ids).logits
        cache = None
        step_logits = []
        for position in range(input_ids.size(1)):
            logits, cache = model.forward_step(
                input_ids[:, position : position + 1],
                cache,
            )
            step_logits.append(logits)
    cached_logits = torch.cat(step_logits, dim=1)
    torch.testing.assert_close(
        full_logits,
        cached_logits,
        atol=2e-5,
        rtol=2e-5,
    )
    assert cache[0]["latent_kv"].shape[1] == input_ids.size(1)
    assert cache[0]["index_keys"].shape[1] == input_ids.size(1)
    assert "index_keys" not in cache[3]


def test_training_monitor_writes_dashboard_artifacts(tmp_path) -> None:
    monitor = TrainingMonitor(
        tmp_path,
        model_parameters=50_153_344,
        active_parameters=21_768_064,
        total_steps=100,
        architecture={"layers": 13},
    )
    monitor.record_train(
        step=0,
        loss=9.4,
        main_loss=9.1,
        mtp_loss=9.0,
        indexer_loss=2.0,
        lr_scale=0.1,
        tokens_per_second=123.0,
        tokens_seen=8192,
        vram_gib=1.2,
        grad_norm=0.9,
        elapsed_seconds=10.0,
        eta_seconds=990.0,
    )
    monitor.record_validation(0, 9.0)
    monitor.record_sample(
        0,
        "Once upon a time",
        "Once upon a time there was a fox.",
    )
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "loss.png").stat().st_size > 1_000
    state = json.loads(
        (tmp_path / "dashboard_state.json").read_text(encoding="utf-8")
    )
    assert state["train_loss"] == 9.4
    assert state["main_loss"] == 9.1
    assert state["sample_text"].endswith("fox.")


def test_safetensors_inference_bundle_round_trip(tmp_path) -> None:
    torch.manual_seed(17)
    model = GLM52ForCausalLM(tiny_config()).eval()
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "model.safetensors"
    config_path.write_text(
        json.dumps(model.config.to_dict()),
        encoding="utf-8",
    )
    save_model(model, weights_path)
    restored = load_trained_model(weights_path, torch.device("cpu"))
    input_ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        expected = model(input_ids).logits
        actual = restored(input_ids).logits
    torch.testing.assert_close(expected, actual)
