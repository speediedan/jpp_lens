"""Tests for the Gemma 3 LRP patchers (``Gemma3RMSNorm``, ``Gemma3MLP``), mirroring
test_lrp_gemma_olmo.py's guarantees on the Gemma 3 text decoder that
``google/gemma-3-1b-it`` loads as:

1. Forward values are unchanged: bit-exact under the detach-only rules
   (LN-rule, half-rule); equal up to kernel dust under the identity rule's
   GELU-tanh rewrite.
2. The patch report covers exactly the residual-stream norms (four per block
   plus the final norm) and every block MLP, and leaves the attention-internal
   q/k norms alone. Without the two registrations the surgery refuses the
   classes by name, so the registrations are what make Gemma 3 fittable.
3. Gradients follow the rules: the LN-rule gives the fixed diagonal scale with
   Gemma 3's ``1 + w`` weight; the GELU identity rule matches a reference
   built from the module's own weights in value and gradient.
4. The norm patcher is bit-exact in bf16 with non-unit weights, where the cast
   order decides the bits, and exiting the context restores stock behaviour.
"""

from __future__ import annotations

import math

import pytest
import torch as t

from workspace_lens.lrp import lrp
from workspace_lens.lrp.lrp import LrpRuleConfig, apply_lrp_rules, lrp_rule_config_for_mode
from workspace_lens.tests.tiny_gemma_olmo import TINY_D_MODEL, build_tiny_gemma3
from workspace_lens.tests.tiny_qwen import last_hidden_state_BSN
from workspace_lens.tests.tiny_qwen import random_tiny_qwen_input_ids as random_tiny_input_ids

GEMMA3_BLOCK_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)
DETACH_ONLY_RULES = LrpRuleConfig(ln_rule=True, identity_rule=False, half_rule=True)
FULL_RULES = lrp_rule_config_for_mode("rlens")


def _with_random_norm_weights(model, seed: int = 11):
    """Gemma 3 zero-initialises its norm weights (``1 + w`` = 1); give every norm a non-zero
    ``w`` so the tests distinguish ``1 + w`` from ``w`` and from no weight at all."""
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for module in model.modules():
            if type(module).__name__ == "Gemma3RMSNorm":
                module.weight.copy_(t.rand(module.weight.shape, generator=generator) * 2 - 0.75)
    return model


@pytest.fixture(scope="module")
def tiny_gemma3():
    return _with_random_norm_weights(build_tiny_gemma3())


@pytest.fixture(scope="module")
def input_ids_Int_1S() -> t.Tensor:
    return random_tiny_input_ids()


### 1. FORWARD VALUES


def test_detach_only_rules_are_bit_exact(tiny_gemma3, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_gemma3, input_ids_Int_1S)
    with apply_lrp_rules(tiny_gemma3, DETACH_ONLY_RULES):
        patched_BSN = last_hidden_state_BSN(tiny_gemma3, input_ids_Int_1S)
    assert t.equal(stock_BSN, patched_BSN)


def test_full_rules_value_close_up_to_activation_kernel_dust(tiny_gemma3, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_gemma3, input_ids_Int_1S)
    with apply_lrp_rules(tiny_gemma3, FULL_RULES):
        patched_BSN = last_hidden_state_BSN(tiny_gemma3, input_ids_Int_1S)
    assert t.allclose(stock_BSN, patched_BSN, rtol=1e-4, atol=1e-5)


def test_context_exit_restores_stock_bit_exact(tiny_gemma3, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_gemma3, input_ids_Int_1S)
    with apply_lrp_rules(tiny_gemma3, FULL_RULES):
        pass
    assert t.equal(stock_BSN, last_hidden_state_BSN(tiny_gemma3, input_ids_Int_1S))
    assert "forward" not in tiny_gemma3.model.layers[0].mlp.__dict__
    assert "forward" not in tiny_gemma3.model.norm.__dict__


### 2. WHICH MODULES ARE PATCHED


def test_patch_report_covers_block_norms_and_mlps_not_attention_norms(tiny_gemma3):
    with apply_lrp_rules(tiny_gemma3, FULL_RULES) as report:
        pass
    num_layers = len(tiny_gemma3.model.layers)
    expected_norms = {"model.norm"} | {
        f"model.layers.{i}.{norm}" for i in range(num_layers) for norm in GEMMA3_BLOCK_NORMS
    }
    assert set(report.ln_rule) == expected_norms
    assert set(report.mlp_rule) == {f"model.layers.{i}.mlp" for i in range(num_layers)}
    attention = tiny_gemma3.model.layers[0].self_attn
    assert type(attention.q_norm).__name__ == type(attention.k_norm).__name__ == "Gemma3RMSNorm"
    assert not any(".self_attn." in name for name in report.ln_rule)


@pytest.mark.parametrize(
    "table, entry, refused_class",
    [
        ("_RMSNORM_LRP_FORWARDS", "Gemma3RMSNorm", "Gemma3RMSNorm"),
        ("_KNOWN_GATED_MLPS", "Gemma3MLP", "Gemma3MLP"),
    ],
)
def test_without_its_registration_each_class_is_refused_by_name(
    tiny_gemma3, monkeypatch, table, entry, refused_class
):
    # The registrations are load-bearing: the walker reaches these classes and, unregistered,
    # refuses them rather than fitting a partly patched lens.
    registered = getattr(lrp, table)
    if isinstance(registered, dict):
        monkeypatch.setattr(lrp, table, {k: v for k, v in registered.items() if k != entry})
    else:
        monkeypatch.setattr(lrp, table, tuple(k for k in registered if k != entry))
    with pytest.raises(RuntimeError, match=refused_class):
        with apply_lrp_rules(tiny_gemma3, FULL_RULES):
            pass


### 3. GRADIENTS


def test_ln_rule_gradient_is_fixed_diagonal_scale_with_one_plus_w(tiny_gemma3):
    # LN-rule: y = x * [rsqrt(mean(x^2) + eps)]detached * (1 + w), so the input gradient
    # must be exactly cotangent * (1 + w) * scale, with no projection term from the
    # variance. Gemma 3 stores w and applies 1 + w (Gemma 4 stores the full scale).
    rmsnorm = tiny_gemma3.model.norm
    generator = t.Generator().manual_seed(2)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)

    with apply_lrp_rules(tiny_gemma3, LrpRuleConfig(ln_rule=True)):
        (patched_grad_SN,) = t.autograd.grad(rmsnorm(x_SN), x_SN, cotangent_SN)

    scale_S1 = t.rsqrt(x_SN.detach().pow(2).mean(-1, keepdim=True) + rmsnorm.eps)
    expected_grad_SN = cotangent_SN * (1.0 + rmsnorm.weight) * scale_S1
    assert t.allclose(patched_grad_SN, expected_grad_SN, rtol=1e-5, atol=1e-7)
    # the plain-weight form would be wrong here, and so would the stock gradient
    assert not t.allclose(patched_grad_SN, cotangent_SN * rmsnorm.weight * scale_S1, rtol=1e-3, atol=1e-5)
    (stock_grad_SN,) = t.autograd.grad(rmsnorm(x_SN), x_SN, cotangent_SN)
    assert not t.allclose(patched_grad_SN, stock_grad_SN, rtol=1e-3, atol=1e-5)


def test_gelu_identity_rule_matches_reference_implementation(tiny_gemma3):
    # Independent reference built from the module's own weights: the patched MLP must equal
    # down((z * gate(z).detach()) * up(x)) in value and gradient, with gate the
    # tanh-approximation factor of gelu_pytorch_tanh.
    mlp = tiny_gemma3.model.layers[0].mlp
    assert type(mlp.act_fn).__name__ == "GELUTanh"
    generator = t.Generator().manual_seed(3)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)

    with apply_lrp_rules(tiny_gemma3, LrpRuleConfig(identity_rule=True)):
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


def test_half_rule_halves_the_mlp_gradient(tiny_gemma3):
    mlp = tiny_gemma3.model.layers[0].mlp
    generator = t.Generator().manual_seed(4)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)
    (stock_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    with apply_lrp_rules(tiny_gemma3, LrpRuleConfig(half_rule=True)):
        (patched_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    assert t.allclose(patched_grad_SN, 0.5 * stock_grad_SN, rtol=1e-6, atol=1e-7)


### 4. BF16


def test_rmsnorm_patcher_is_bit_exact_in_bf16_with_non_unit_weights(tiny_gemma3):
    # gemma-3-1b-it runs in bf16, where normalising in fp32, applying 1 + w in fp32 and only
    # then casting decides the bits. A wrong order would still pass the fp32 tests above.
    rmsnorm = tiny_gemma3.model.layers[0].post_attention_layernorm
    generator = t.Generator().manual_seed(7)
    x_SN = (t.randn(9, TINY_D_MODEL, generator=generator) * 4).to(t.bfloat16)
    stock_SN = rmsnorm(x_SN)
    with apply_lrp_rules(tiny_gemma3, LrpRuleConfig(ln_rule=True)):
        patched_SN = rmsnorm(x_SN)
    assert patched_SN.dtype == t.bfloat16 and t.equal(stock_SN, patched_SN)
