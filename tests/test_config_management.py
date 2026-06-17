import numpy as np
import pytest

from hypersonic_rl.envs import PursueEscapeEnv, PursueEscapeEnvConfig
from hypersonic_rl.utils import build_dataclass_config


def test_build_dataclass_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="bad_parameter_name"):
        build_dataclass_config(
            {
                "dt": 0.01,
                "bad_parameter_name": 1.0,
            },
            PursueEscapeEnvConfig,
        )


def test_custom_ability_profile_preserves_explicit_overloads():
    env_config = build_dataclass_config(
        {
            "interceptor_ability_profile": "custom",
            "source_pn_max_overload": 5.5,
            "interceptor_max_overload": 7.5,
            "initial_randomization_enabled": False,
        },
        PursueEscapeEnvConfig,
    )

    env = PursueEscapeEnv(env_config)

    assert np.isclose(env.config.source_pn_max_overload, 5.5)
    assert np.isclose(env.config.interceptor_max_overload, 7.5)
    assert np.isclose(env.navigation_config.max_overload, 5.5)
    assert np.isclose(env.interceptor_config.max_overload, 7.5)


def test_named_ability_profile_still_applies_profile_overload():
    env = PursueEscapeEnv(
        PursueEscapeEnvConfig(
            interceptor_ability_profile="strong",
            initial_randomization_enabled=False,
        )
    )

    assert np.isclose(env.config.source_pn_max_overload, 12.0)
    assert np.isclose(env.config.interceptor_max_overload, 12.0)
