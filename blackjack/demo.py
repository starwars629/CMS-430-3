import random
from env import BlackjackEnv


def random_agent(state: tuple[int, int, int]) -> int:
    return random.randint(0, 1)


def threshold_agent(state: tuple[int, int, int], threshold: int = 17) -> int:
    player_value, _, _ = state
    return 0 if player_value >= threshold else 1


def run_simulation(agent_fn, n_episodes: int = 10_000) -> float:
    env = BlackjackEnv()
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        reward = 0.0
        while not done:
            action = agent_fn(state)
            state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward / n_episodes


if __name__ == "__main__":
    n = 10_000

    print(f"Running {n:,} episodes per agent...\n")

    random_rate = run_simulation(random_agent, n_episodes=n)
    print(f"Random agent win rate:      {random_rate:.4f} ({random_rate*100:.2f}%)")

    threshold_rate = run_simulation(threshold_agent, n_episodes=n)
    print(f"Threshold-17 agent win rate: {threshold_rate:.4f} ({threshold_rate*100:.2f}%)")
