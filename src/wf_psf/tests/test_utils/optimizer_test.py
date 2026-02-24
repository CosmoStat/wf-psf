import pytest
from types import SimpleNamespace
from wf_psf.utils.optimizer import get_optimizer
from wf_psf.utils.read_config import RecursiveNamespace


# Dummy optimizer classes
class DummyAdam:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = "Adam"
        self.beta_1 = kwargs.get("beta_1")
        self.beta_2 = kwargs.get("beta_2")
        self.epsilon = kwargs.get("epsilon")
        self.amsgrad = kwargs.get("amsgrad")
        self.learning_rate = kwargs.get("learning_rate")


class DummyLegacyAdam(DummyAdam):
    pass


class DummyRAdam:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._name = "RectifiedAdam"
        self.learning_rate = kwargs.get("learning_rate")


# Parametric test for Adam with overrides
@pytest.mark.parametrize(
    "optimizer_input, expected_lr, expected_beta1",
    [
        ("adam", 0.001, 0.9),
        ({"name": "adam", "learning_rate": 0.01, "beta_1": 0.8}, 0.01, 0.8),
        (RecursiveNamespace(name="adam", learning_rate=0.02, beta_1=0.85), 0.02, 0.85),
    ],
)
def test_adam_optimizer_overrides(
    monkeypatch, optimizer_input, expected_lr, expected_beta1
):
    # Mock TF >= 2.11
    fake_tf = SimpleNamespace(
        __version__="2.11.0",
        keras=SimpleNamespace(
            optimizers=SimpleNamespace(
                Adam=DummyAdam, legacy=SimpleNamespace(Adam=DummyLegacyAdam)
            )
        ),
    )
    monkeypatch.setattr("wf_psf.utils.optimizer.tf", fake_tf)

    opt = get_optimizer(optimizer_input)
    assert isinstance(opt, DummyAdam)
    assert opt.name.lower() == "adam"
    assert opt.learning_rate == expected_lr
    assert opt.beta_1 == expected_beta1


# Parametric test for RAdam with overrides
@pytest.mark.parametrize(
    "optimizer_input, expected_lr",
    [
        ("rectified_adam", 0.001),
        ({"name": "rectified_adam", "learning_rate": 0.01}, 0.01),
        (RecursiveNamespace(name="rectified_adam", learning_rate=0.02), 0.02),
    ],
)
def test_radam_optimizer_overrides(monkeypatch, optimizer_input, expected_lr):
    # Provide dummy tfa module
    dummy_tfa = SimpleNamespace(optimizers=SimpleNamespace(RectifiedAdam=DummyRAdam))
    monkeypatch.setitem(__import__("sys").modules, "tensorflow_addons", dummy_tfa)

    opt = get_optimizer(optimizer_input)
    assert isinstance(opt, DummyRAdam)
    assert opt._name.lower() == "rectifiedadam"
    assert opt.learning_rate == expected_lr


def test_legacy_adam_handling(monkeypatch):
    """Verify that legacy.Adam is used when TF < 2.11 and parameters are applied correctly."""

    # Mock TF < 2.11
    fake_tf = SimpleNamespace(
        __version__="2.10.0",
        keras=SimpleNamespace(
            optimizers=SimpleNamespace(
                Adam=DummyAdam, legacy=SimpleNamespace(Adam=DummyLegacyAdam)
            )
        ),
    )
    monkeypatch.setattr("wf_psf.utils.optimizer.tf", fake_tf)

    # Provide RecursiveNamespace input with overrides
    opt_config = RecursiveNamespace(
        name="adam",
        learning_rate=0.02,
        beta_1=0.85,
        beta_2=0.95,
        epsilon=1e-08,
        amsgrad=True,
    )

    opt = get_optimizer(opt_config)
    assert isinstance(opt, DummyLegacyAdam)
    assert opt.name.lower() == "adam"
    assert opt.learning_rate == 0.02
    assert opt.beta_1 == 0.85
    assert opt.beta_2 == 0.95
    assert opt.epsilon == 1e-08
    assert opt.amsgrad is True
