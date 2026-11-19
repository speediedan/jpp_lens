"""Tests for the Gemma 2 LRP patchers (``Gemma2RMSNorm``, ``Gemma2MLP``) on the text model
``google/gemma-2-2b`` loads as, mirroring test_lrp_gemma3.py. Gemma 2's two classes have
the same forwards as Gemma 3's, so the guarantees are the same, plus the two places Gemma 2
differs:

1. Forward values are unchanged: bit-exact under the detach-only rules; equal up to kernel
   dust under the identity rule's GELU-tanh rewrite. Exiting the context restores stock.
2. The patch report covers the four block norms, the final norm and every MLP. Gemma 2's
   attention has no internal norms (Gemma 3 adds q/k norms), and its attention-logit soft cap
   stays inside ``self_attn``, which keeps standard gradients. Without either registration the
   surgery refuses the class by name.
3. Gradients follow the rules: the LN-rule's fixed diagonal scale uses ``1 + w``; the GELU
   identity rule matches an independent reference; the half rule halves the MLP gradient.
4. The norm patcher is bit-exact in bf16 with non-unit weights.
5. The readout applies Gemma 2's final-logit soft cap (the LensModel's ``unembed`` reads it from
   the config), so a fit's logits are the model's own.
"""

from __future__ import annotations

import math

import pytest
import torch as t

import jlens
from workspace_lens.lrp import lrp
from workspace_lens.lrp.lrp import LrpRuleConfig, apply_lrp_rules, lrp_rule_config_for_mode
from workspace_lens.tests.tiny_gemma_olmo import TINY_D_MODEL, build_tiny_gemma2
from workspace_lens.tests.tiny_qwen import ByteVocabTokenizer, last_hidden_state_BSN
from workspace_lens.tests.tiny_qwen import random_tiny_qwen_input_ids as random_tiny_input_ids

GEMMA2_BLOCK_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)
DETACH_ONLY_RULES = LrpRuleConfig(ln_rule=True, identity_rule=False, half_rule=True)
FULL_RULES = lrp_rule_config_for_mode("rlens")


def _with_random_norm_weights(model, seed: int = 11):
    """Gemma 2 zero-initialises its norm weights (``1 + w`` = 1); give every norm a non-zero
    ``w`` so the tests distinguish ``1 + w`` from ``w`` and from no weight at all."""
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for module in model.modules():
            if type(module).__name__ == "Gemma2RMSNorm":
                module.weight.copy_(t.rand(module.weight.shape, generator=generator) * 2 - 0.75)
    return model


@pytest.fixture(scope="module")
def tiny_gemma2():
    return _with_random_norm_weights(build_tiny_gemma2())


@pytest.fixture(scope="module")
def input_ids_Int_1S() -> t.Tensor:
    return random_tiny_input_ids()


### 1. FORWARD VALUES


def test_detach_only_rules_are_bit_exact(tiny_gemma2, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_gemma2, input_ids_Int_1S)
    with apply_lrp_rules(tiny_gemma2, DETACH_ONLY_RULES):
        patched_BSN = last_hidden_state_BSN(tiny_gemma2, input_ids_Int_1S)
    assert t.equal(stock_BSN, patched_BSN)


def test_full_rules_value_close_up_to_activation_kernel_dust(tiny_gemma2, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_gemma2, input_ids_Int_1S)
    with apply_lrp_rules(tiny_gemma2, FULL_RULES):
        patched_BSN = last_hidden_state_BSN(tiny_gemma2, input_ids_Int_1S)
    assert t.allclose(stock_BSN, patched_BSN, rtol=1e-4, atol=1e-5)


def test_context_exit_restores_stock_bit_exact(tiny_gemma2, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_gemma2, input_ids_Int_1S)
    with apply_lrp_rules(tiny_gemma2, FULL_RULES):
        pass
    assert t.equal(stock_BSN, last_hidden_state_BSN(tiny_gemma2, input_ids_Int_1S))
    assert "forward" not in tiny_gemma2.model.layers[0].mlp.__dict__
    assert "forward" not in tiny_gemma2.model.norm.__dict__


### 2. WHICH MODULES ARE PATCHED


def test_patch_report_covers_block_norms_and_mlps(tiny_gemma2):
    with apply_lrp_rules(tiny_gemma2, FULL_RULES) as report:
        pass
    num_layers = len(tiny_gemma2.model.layers)
    expected_norms = {"model.norm"} | {
        f"model.layers.{i}.{norm}" for i in range(num_layers) for norm in GEMMA2_BLOCK_NORMS
    }
    assert set(report.ln_rule) == expected_norms
    assert set(report.mlp_rule) == {f"model.layers.{i}.mlp" for i in range(num_layers)}
    attention = tiny_gemma2.model.layers[0].self_attn
    # no norms inside Gemma 2's attention; its soft cap is a config scalar applied there
    assert not any(type(m).__name__ == "Gemma2RMSNorm" for m in attention.modules())
    assert attention.attn_logit_softcapping == 50.0
    assert not any(".self_attn" in name for name in [*report.ln_rule, *report.mlp_rule])


@pytest.mark.parametrize(
    "table, entry",
    [("_RMSNORM_LRP_FORWARDS", "Gemma2RMSNorm"), ("_KNOWN_GATED_MLPS", "Gemma2MLP")],
)
def test_without_its_registration_each_class_is_refused_by_name(tiny_gemma2, monkeypatch, table, entry):
    registered = getattr(lrp, table)
    if isinstance(registered, dict):
        monkeypatch.setattr(lrp, table, {k: v for k, v in registered.items() if k != entry})
    else:
        monkeypatch.setattr(lrp, table, tuple(k for k in registered if k != entry))
    with pytest.raises(RuntimeError, match=entry):
        with apply_lrp_rules(tiny_gemma2, FULL_RULES):
            pass


### 3. GRADIENTS


def test_ln_rule_gradient_is_fixed_diagonal_scale_with_one_plus_w(tiny_gemma2):
    rmsnorm = tiny_gemma2.model.norm
    generator = t.Generator().manual_seed(2)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)

    with apply_lrp_rules(tiny_gemma2, LrpRuleConfig(ln_rule=True)):
        (patched_grad_SN,) = t.autograd.grad(rmsnorm(x_SN), x_SN, cotangent_SN)

    scale_S1 = t.rsqrt(x_SN.detach().pow(2).mean(-1, keepdim=True) + rmsnorm.eps)
    expected_grad_SN = cotangent_SN * (1.0 + rmsnorm.weight) * scale_S1
    assert t.allclose(patched_grad_SN, expected_grad_SN, rtol=1e-5, atol=1e-7)
    assert not t.allclose(patched_grad_SN, cotangent_SN * rmsnorm.weight * scale_S1, rtol=1e-3, atol=1e-5)
    (stock_grad_SN,) = t.autograd.grad(rmsnorm(x_SN), x_SN, cotangent_SN)
    assert not t.allclose(patched_grad_SN, stock_grad_SN, rtol=1e-3, atol=1e-5)


def test_gelu_identity_rule_matches_reference_implementation(tiny_gemma2):
    mlp = tiny_gemma2.model.layers[0].mlp
    assert type(mlp.act_fn).__name__ == "GELUTanh"
    generator = t.Generator().manual_seed(3)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)

    with apply_lrp_rules(tiny_gemma2, LrpRuleConfig(identity_rule=True)):
        patched_SN = mlp(x_SN)
        (patched_grad_SN,) = t.autograd.grad(patched_SN, x_SN, cotangent_SN)

    z_SI = mlp.gate_proj(x_SN)
    gate_SI = 0.5 * (1.0 + t.tanh(math.sqrt(2.0 / math.pi) * (z_SI + 0.044715 * z_SI.pow(3))))
    reference_SN = mlp.down_proj((z_SI * gate_SI.detach()) * mlp.up_proj(x_SN))
    (reference_grad_SN,) = t.autograd.grad(reference_SN, x_SN, cotangent_SN)
    assert t.allclose(patched_SN, reference_SN, rtol=1e-5, atol=1e-6)
    assert t.allclose(patched_grad_SN, reference_grad_SN, rtol=1e-5, atol=1e-6)

    (stock_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    assert not t.allclose(patched_grad_SN, stock_grad_SN, rtol=1e-3, atol=1e-5)


def test_half_rule_halves_the_mlp_gradient(tiny_gemma2):
    mlp = tiny_gemma2.model.layers[0].mlp
    generator = t.Generator().manual_seed(4)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)
    (stock_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    with apply_lrp_rules(tiny_gemma2, LrpRuleConfig(half_rule=True)):
        (patched_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    assert t.allclose(patched_grad_SN, 0.5 * stock_grad_SN, rtol=1e-6, atol=1e-7)


### 4. BF16


def test_rmsnorm_patcher_is_bit_exact_in_bf16_with_non_unit_weights(tiny_gemma2):
    rmsnorm = tiny_gemma2.model.layers[0].post_attention_layernorm
    generator = t.Generator().manual_seed(7)
    x_SN = (t.randn(9, TINY_D_MODEL, generator=generator) * 4).to(t.bfloat16)
    stock_SN = rmsnorm(x_SN)
    with apply_lrp_rules(tiny_gemma2, LrpRuleConfig(ln_rule=True)):
        patched_SN = rmsnorm(x_SN)
    assert patched_SN.dtype == t.bfloat16 and t.equal(stock_SN, patched_SN)


### 5. THE READOUT'S SOFT CAP


def test_unembed_applies_the_final_logit_soft_cap(tiny_gemma2):
    # The fit's targets are the LensModel's logits, so they must be the model's own, soft cap included.
    lens_model = jlens.from_hf(tiny_gemma2, ByteVocabTokenizer())
    generator = t.Generator().manual_seed(5)
    residual_SN = t.randn(6, TINY_D_MODEL, generator=generator)
    with t.no_grad():
        uncapped_SV = tiny_gemma2.lm_head(tiny_gemma2.model.norm(residual_SN))
        # scale the head so some logits exceed the cap: a cap that never binds would pass untested
        scale = 60.0 / float(uncapped_SV.abs().max())
        tiny_gemma2.lm_head.weight.mul_(scale)
        try:
            uncapped_SV = tiny_gemma2.lm_head(tiny_gemma2.model.norm(residual_SN))
            capped_SV = lens_model.unembed(residual_SN)
        finally:
            tiny_gemma2.lm_head.weight.div_(scale)
    assert uncapped_SV.abs().max() > 30.0
    assert t.allclose(capped_SV, 30.0 * t.tanh(uncapped_SV / 30.0), rtol=1e-5, atol=1e-5)
