from hypersonic_rl.envs.pursue_escape_env import PursueEscapeEnv, PursueEscapeEnvConfig


def run_mode(guidance_mode: str) -> None:
    config = PursueEscapeEnvConfig(
        guidance_mode=guidance_mode,
        initial_randomization_enabled=False,
        interceptor_count=2,
        t=2.0,
        dt=0.01,
        paper_midcourse_time_scale=2.0,
        paper_midcourse_navigation_gain=6.0,
        paper_midcourse_theta_bias_gain=0.07,
        paper_terminal_navigation_gain=6.0,
    )

    env = PursueEscapeEnv(config=config)
    obs, info = env.reset(seed=0)

    last_info = info
    for _ in range(50):
        action = env.action_space.sample() * 0.0
        obs, reward, terminated, truncated, last_info = env.step(action)

        if terminated or truncated:
            break

    print("=" * 80)
    print(f"guidance_mode = {guidance_mode}")
    print(f"terminated = {terminated}, truncated = {truncated}, reward = {reward}")

    interesting_keys = [
        key for key in sorted(last_info.keys())
        if (
            "phase" in key
            or "paper_midcourse" in key
            or "midcourse" in key
            or "terminal" in key
            or "source_" in key
            or "saturation" in key
        )
    ]

    for key in interesting_keys[:80]:
        print(f"{key}: {last_info[key]}")


if __name__ == "__main__":
    for mode in [
        "source_pn",
        "mid_terminal_interceptor",
        "paper_mid_terminal",
    ]:
        run_mode(mode)