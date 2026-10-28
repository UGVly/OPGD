"""CPU contracts for the entropy-weighted GRPO recipe."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_entropy_token_weights
from verl.workers.config.actor import PolicyLossConfig

VERL_ROOT = Path(__file__).resolve().parents[3]


def _mask(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_none_mode_returns_none():
    entropy = torch.rand(2, 4)
    mask = _mask([[1, 1, 1, 1], [1, 1, 1, 1]])
    assert compute_entropy_token_weights(entropy, mask, mode="none") is None


def test_low_entropy_token_gets_weight_one():
    # Entropy increases left to right, so -entropy decreases left to right.
    entropy = torch.tensor([[0.1, 0.5, 1.0, 2.0]])
    mask = _mask([[1, 1, 1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")

    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, -1] == pytest.approx(0.0)
    assert torch.all(weights[0, :-1] >= weights[0, 1:]), "weights must fall as entropy rises"


def test_weights_stay_in_unit_interval():
    entropy = torch.rand(8, 32) * 5.0
    mask = (_mask([[1] * 32] * 8)).to(torch.float32)

    weights = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")

    assert weights.min().item() >= 0.0
    assert weights.max().item() <= 1.0


def test_response_scope_normalizes_per_response():
    entropy = torch.tensor([[0.0, 1.0], [10.0, 11.0]])
    mask = _mask([[1, 1], [1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")

    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(0.0)
    assert weights[1, 0] == pytest.approx(1.0)
    assert weights[1, 1] == pytest.approx(0.0)


def test_batch_scope_normalizes_over_the_batch():
    entropy = torch.tensor([[0.0, 1.0], [10.0, 11.0]])
    mask = _mask([[1, 1], [1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="batch")

    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[1, 1] == pytest.approx(0.0)


def test_padded_positions_are_zeroed():
    entropy = torch.tensor([[0.0, 1.0, 5.0]])
    mask = _mask([[1, 1, 0]])

    weights = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")

    assert weights[0, 2] == pytest.approx(0.0)


def test_constant_entropy_response_keeps_unit_weight():
    entropy = torch.full((1, 3), 0.7)
    mask = _mask([[1, 1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")

    assert torch.allclose(weights, torch.ones(1, 3))


def test_renormalize_mean_gives_unit_mean_and_keeps_direction():
    entropy = torch.tensor([[0.0, 0.4, 0.8, 1.2]])
    mask = _mask([[1, 1, 1, 1]])

    raw = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")
    renorm = compute_entropy_token_weights(
        entropy, mask, mode="neg_entropy_minmax", scope="response", renormalize_mean=True
    )

    assert renorm[mask.to(torch.bool)].mean().item() == pytest.approx(1.0)
    # Renormalization is a positive global rescale, so it preserves the relative ordering.
    assert torch.allclose(renorm / renorm.sum(), raw / raw.sum())


def test_minmax_raw_mean_is_below_one():
    """Documents the scale caveat: raw min-max weights average well below 1."""
    entropy = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = _mask([[1, 1, 1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")

    assert weights.mean().item() == pytest.approx(0.5)


def test_unknown_mode_and_scope_raise():
    entropy = torch.rand(1, 2)
    mask = _mask([[1, 1]])

    with pytest.raises(ValueError):
        compute_entropy_token_weights(entropy, mask, mode="bogus")
    with pytest.raises(ValueError):
        compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="bogus")


def test_weighting_composes_with_advantage_like_the_actor_does():
    advantages = torch.tensor([[2.0, -4.0, 1.0, 8.0]])
    entropy = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = _mask([[1, 1, 1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")
    weighted = advantages * weights

    # The most uncertain token carries the largest raw advantage here, so it is muted.
    assert weighted[0, -1] == pytest.approx(0.0)
    assert weighted[0, 0] == pytest.approx(2.0)


def test_pos_mode_mirrors_neg_mode():
    entropy = torch.tensor([[0.1, 0.5, 1.0, 2.0]])
    mask = _mask([[1, 1, 1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="pos_entropy_minmax", scope="response")

    assert weights[0, 0] == pytest.approx(0.0)
    assert weights[0, -1] == pytest.approx(1.0)
    assert torch.all(weights[0, :-1] <= weights[0, 1:]), "weights must rise with entropy"


def test_pos_mode_is_the_complement_of_neg_mode():
    entropy = torch.rand(4, 16) * 4.0
    mask = _mask([[1] * 16] * 4)

    neg = compute_entropy_token_weights(entropy, mask, mode="neg_entropy_minmax", scope="response")
    pos = compute_entropy_token_weights(entropy, mask, mode="pos_entropy_minmax", scope="response")

    # Both modes min-max normalize, so the two sets of weights sum to one on valid tokens.
    assert torch.allclose(neg + pos, torch.ones_like(neg), atol=1e-6)


def test_pos_mode_emphasizes_the_largest_advantage_token():
    advantages = torch.tensor([[2.0, -4.0, 1.0, 8.0]])
    entropy = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = _mask([[1, 1, 1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="pos_entropy_minmax", scope="response")
    weighted = advantages * weights

    assert weighted[0, 0] == pytest.approx(0.0)
    assert weighted[0, -1] == pytest.approx(8.0)


def test_pos_mode_constant_entropy_keeps_unit_weight():
    entropy = torch.full((1, 3), 0.7)
    mask = _mask([[1, 1, 1]])

    weights = compute_entropy_token_weights(entropy, mask, mode="pos_entropy_minmax", scope="response")

    assert torch.allclose(weights, torch.ones(1, 3))


def test_pos_mode_zeroes_padded_positions():
    entropy = torch.tensor([[0.0, 1.0, 5.0]])
    mask = _mask([[1, 1, 0]])

    weights = compute_entropy_token_weights(entropy, mask, mode="pos_entropy_minmax", scope="response")

    assert weights[0, 2] == pytest.approx(0.0)


def test_correlation_accumulator_matches_a_direct_computation():
    from verl.workers.actor.dp_actor import _CorrelationAccumulator

    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

    acc = _CorrelationAccumulator()
    acc.update(x, y)

    assert acc.corr() == pytest.approx(1.0)
    assert acc.slope() == pytest.approx(2.0)

    neg = _CorrelationAccumulator()
    neg.update(x, -y)
    assert neg.corr() == pytest.approx(-1.0)
    assert neg.slope() == pytest.approx(-2.0)


def test_correlation_accumulator_matches_torch_corrcoef_when_chunked():
    from verl.workers.actor.dp_actor import _CorrelationAccumulator

    torch.manual_seed(0)
    x = torch.randn(500) * 1.7 + 0.4
    y = 0.3 * x + torch.randn(500)

    acc = _CorrelationAccumulator()
    for start in range(0, 500, 37):  # uneven chunks, as dynamic batching produces
        acc.update(x[start : start + 37], y[start : start + 37])

    reference = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
    assert acc.corr() == pytest.approx(reference, abs=1e-5)
    # The slope is the least-squares coefficient, which for this generating process is near 0.3.
    assert acc.slope() == pytest.approx(0.3, abs=0.1)


def test_correlation_accumulator_reports_neutral_values_without_variance():
    from verl.workers.actor.dp_actor import _CorrelationAccumulator

    empty = _CorrelationAccumulator()
    assert empty.corr() == 0.0
    assert empty.slope() == 0.0

    # A constant advantage, which is what a single-response micro-batch looks like under GRPO,
    # must not be reported as a correlation.
    constant = _CorrelationAccumulator()
    constant.update(torch.tensor([0.1, 0.5, 2.0]), torch.full((3,), 1.25))
    assert constant.corr() == 0.0
    assert constant.slope() == 0.0

    # Mismatched lengths and empty input are ignored rather than corrupting the sums.
    mismatched = _CorrelationAccumulator()
    mismatched.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))
    assert mismatched.corr() == 0.0


def test_default_config_disables_weighting():
    config = OmegaConf.load(VERL_ROOT / "verl/trainer/config/actor/actor.yaml")
    policy_loss = config.policy_loss

    assert policy_loss.entropy_token_weight_mode == "none"
    assert policy_loss.entropy_token_weight_scope == "response"
    assert policy_loss.entropy_token_weight_renormalize_mean is False


def test_policy_loss_config_defaults_match_the_yaml_contract():
    policy_loss = PolicyLossConfig()

    assert policy_loss.entropy_token_weight_mode == "none"
    assert policy_loss.entropy_token_weight_scope == "response"
    assert policy_loss.entropy_token_weight_renormalize_mean is False


def test_recipe_config_enables_response_scoped_weighting():
    config = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/entweight_trainer.yaml")
    policy_loss = config.actor_rollout_ref.actor.policy_loss

    # The recipe emphasizes the most uncertain tokens and rescales the weights to mean 1,
    # because the raw pos_entropy_minmax weights average about 0.11 and would shrink the
    # update to roughly a ninth of GRPO at the same learning rate.
    assert policy_loss.entropy_token_weight_mode == "pos_entropy_minmax"
    assert policy_loss.entropy_token_weight_scope == "response"
    assert policy_loss.entropy_token_weight_renormalize_mean is True
