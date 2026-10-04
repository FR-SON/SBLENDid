import pytest


def test_registry_types_import():
    from src.Benchmark.plans.registry import (
        ArmResult, PlanContext, PlanResult, PlanSpec, PLANS, register,  # noqa: F401
    )
    assert isinstance(PLANS, dict)
    assert hasattr(PlanSpec, "__dataclass_fields__")


def test_register_and_duplicate_rejected():
    from src.Benchmark.plans.registry import PlanSpec, PLANS, register
    import copy
    saved = copy.copy(PLANS)
    try:
        spec = PlanSpec(name="t_dup", description="d", point="p",
                        run=lambda ctx: None)
        register(spec)
        assert PLANS["t_dup"] is spec
        with pytest.raises(ValueError, match="already registered"):
            register(spec)
    finally:
        PLANS.clear()
        PLANS.update(saved)


def test_package_sets_threading_env():
    import os
    import src.Benchmark.plans  # noqa: F401
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert os.environ["OMP_NUM_THREADS"] == "1"
