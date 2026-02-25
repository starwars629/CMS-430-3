"""Genetic Algorithm Blackjack Solver.

Evolves a population of 260-bit chromosomes encoding complete blackjack
strategies (hard hands 4-20, soft hands 12-20, dealer upcards 2-A).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from env import BlackjackEnv

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
POP_SIZE = 100
N_GENERATIONS = 100
N_EPISODES = 1000
MUTATION_RATE = 0.01
N_ELITE = 2
CHROM_LEN = 260  # 170 hard + 90 soft

# Dealer upcard → column index: 2→0, 3→1, …, 10→8, Ace(11)→9
_DEALER_IDX = {v: v - 2 for v in range(2, 11)}
_DEALER_IDX[11] = 9  # Ace


# --------------------------------------------------------------------------- #
# Chromosome helpers
# --------------------------------------------------------------------------- #
def chrom_index(player_val: int, dealer_val: int, is_soft: bool):
    """Return the bit position in the chromosome for this state.

    Returns None when player_val == 21 (always stand, no lookup needed).
    """
    if player_val == 21:
        return None

    dealer_idx = _DEALER_IDX.get(dealer_val)
    if dealer_idx is None:
        return None

    if is_soft:
        # Soft 12–20 → rows 0–8
        if not (12 <= player_val <= 20):
            return None
        return 170 + (player_val - 12) * 10 + dealer_idx
    else:
        # Hard 4–20 → rows 0–16
        if not (4 <= player_val <= 20):
            return None
        return (player_val - 4) * 10 + dealer_idx


def make_agent(chrom: np.ndarray):
    """Wrap a chromosome into an agent callable (state) -> action."""
    def agent(state):
        player_val, dealer_val, is_soft_int = state
        is_soft = bool(is_soft_int)
        idx = chrom_index(player_val, dealer_val, is_soft)
        if idx is None:
            return 0  # stand
        return int(chrom[idx])
    return agent


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def evaluate_fitness(chrom: np.ndarray, n_episodes: int = N_EPISODES) -> float:
    """Simulate n_episodes hands and return average reward."""
    env = BlackjackEnv()
    agent = make_agent(chrom)
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        reward = 0.0
        while not done:
            action = agent(state)
            state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward / n_episodes


# --------------------------------------------------------------------------- #
# GA operators
# --------------------------------------------------------------------------- #
def init_population() -> np.ndarray:
    """Random binary population of shape (POP_SIZE, CHROM_LEN)."""
    return np.random.randint(0, 2, size=(POP_SIZE, CHROM_LEN), dtype=np.int8)


def roulette_select(population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
    """Select one individual via fitness-proportionate (roulette) selection."""
    total = fitnesses.sum()
    if total == 0:
        probs = np.ones(len(fitnesses)) / len(fitnesses)
    else:
        probs = fitnesses / total
    idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()


def crossover(p1: np.ndarray, p2: np.ndarray):
    """Single-point crossover; returns two children."""
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def mutate(chrom: np.ndarray, rate: float = MUTATION_RATE) -> np.ndarray:
    """Flip each bit independently with probability `rate`."""
    mask = np.random.random(CHROM_LEN) < rate
    return np.where(mask, 1 - chrom, chrom).astype(np.int8)


# --------------------------------------------------------------------------- #
# Main GA loop
# --------------------------------------------------------------------------- #
def run_ga():
    """Run the GA; return (final_population, history).

    history is a list of (min, max, median, mean) per generation.
    """
    population = init_population()
    history = []

    for gen in range(N_GENERATIONS):
        # Evaluate fitness for every individual
        fitnesses = np.array([evaluate_fitness(ind) for ind in population])

        # Record stats
        stats = (
            float(fitnesses.min()),
            float(fitnesses.max()),
            float(np.median(fitnesses)),
            float(fitnesses.mean()),
        )
        history.append(stats)
        print(
            f"Gen {gen + 1:3d}/{N_GENERATIONS}  "
            f"min={stats[0]:.4f}  max={stats[1]:.4f}  "
            f"med={stats[2]:.4f}  mean={stats[3]:.4f}"
        )

        # Sort descending by fitness
        order = np.argsort(fitnesses)[::-1]
        population = population[order]
        fitnesses = fitnesses[order]

        # Build next generation
        next_pop = []

        # Elitism: carry top N_ELITE unchanged
        for i in range(N_ELITE):
            next_pop.append(population[i].copy())

        # Fill the rest via selection, crossover, mutation
        while len(next_pop) < POP_SIZE:
            p1 = roulette_select(population, fitnesses)
            p2 = roulette_select(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            next_pop.append(mutate(c1))
            if len(next_pop) < POP_SIZE:
                next_pop.append(mutate(c2))

        population = np.array(next_pop, dtype=np.int8)

    return population, history


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def plot_fitness(history):
    """Line plot of min/max/median/mean vs generation; saves fitness_history.png."""
    gens = range(1, len(history) + 1)
    mins, maxs, meds, means = zip(*history)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, maxs, label="Max", linewidth=2)
    ax.plot(gens, means, label="Mean", linewidth=2, linestyle="--")
    ax.plot(gens, meds, label="Median", linewidth=2, linestyle="-.")
    ax.plot(gens, mins, label="Min", linewidth=2, linestyle=":")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (avg reward)")
    ax.set_title("GA Convergence — Blackjack Strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fitness_history.png", dpi=150)
    plt.close(fig)
    print("Saved fitness_history.png")


def plot_strategy_heatmap(population: np.ndarray):
    """Two side-by-side heatmaps showing % of population recommending Hit.

    Hard hands (rows=player totals 4-20, cols=dealer 2-A).
    Soft hands (rows=player totals soft 12-20, cols=dealer 2-A).
    Saves strategy_heatmap.png.
    """
    dealer_labels = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

    # Build matrices: mean across population (= fraction recommending Hit)
    hard_rows = list(range(4, 21))   # 17 values
    soft_rows = list(range(12, 21))  # 9 values

    hard_matrix = np.zeros((len(hard_rows), 10))
    for r, pv in enumerate(hard_rows):
        for c, dv in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
            idx = chrom_index(pv, dv, is_soft=False)
            if idx is not None:
                hard_matrix[r, c] = population[:, idx].mean()
            # idx None means player_val==21 → always stand → 0

    soft_matrix = np.zeros((len(soft_rows), 10))
    for r, pv in enumerate(soft_rows):
        for c, dv in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
            idx = chrom_index(pv, dv, is_soft=True)
            if idx is not None:
                soft_matrix[r, c] = population[:, idx].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    cmap = "RdBu_r"

    for ax, matrix, row_labels, title in [
        (axes[0], hard_matrix, [str(v) for v in hard_rows], "Hard Hands"),
        (axes[1], soft_matrix, [f"A+{v-11}" for v in soft_rows], "Soft Hands"),
    ]:
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(10))
        ax.set_xticklabels(dealer_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Dealer Upcard")
        ax.set_ylabel("Player Total")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="% Recommending Hit")

    fig.suptitle("Evolved Blackjack Strategy (Blue=Stand, Red=Hit)", fontsize=13)
    plt.tight_layout()
    plt.savefig("strategy_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved strategy_heatmap.png")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(f"Starting GA: pop={POP_SIZE}, generations={N_GENERATIONS}, "
          f"episodes/eval={N_EPISODES}, mutation={MUTATION_RATE}/bit\n")

    final_pop, history = run_ga()

    print("\nPlotting results...")
    plot_fitness(history)
    plot_strategy_heatmap(final_pop)

    best_fitness = max(s[1] for s in history)
    print(f"\nDone. Best fitness observed: {best_fitness:.4f}")
