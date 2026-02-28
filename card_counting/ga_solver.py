"""Genetic Algorithm — Card Counting Blackjack Solver.

Evolves a population of 294-bit chromosomes encoding:
  - Bits   0–259 (260 bits): play strategy (hard 4-20 × dealer 2-A, soft 12-20 × dealer 2-A)
  - Bits 260–281  (22 bits): Hi-Lo-style count values (11 card groups × 2 bits)
  - Bits 282–293  (12 bits): bet multipliers (4 true-count ranges × 3 bits → 1-8×)

Fitness = final bankroll after 1000 hands starting at $1,000.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports resolve when run directly from card_counting/
sys.path.insert(0, os.path.dirname(__file__))

from env import CardCountingEnv, BASE_BET

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
POP_SIZE = 100
N_GENERATIONS = 150
N_HANDS = 1000
MUTATION_RATE = 0.005
N_ELITE = 3
CHROM_LEN = 294   # 260 play + 22 count + 12 bet

# Dealer upcard → column index: 2→0, 3→1, …, 10→8, Ace(11)→9
_DEALER_IDX = {v: v - 2 for v in range(2, 11)}
_DEALER_IDX[11] = 9  # Ace

# Card group names (order matches _COUNT_GROUP in card.py)
_GROUP_NAMES = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Face(J/Q/K)"]

# Hi-Lo reference values per group
_HILO_REF = [-1, +1, +1, +1, +1, +1, 0, 0, 0, -1, -1]

# True-count range labels
_TC_RANGE_LABELS = ["≤ −2", "−1 to +1", "+2 to +4", "≥ +5"]

# --------------------------------------------------------------------------- #
# Chromosome helpers
# --------------------------------------------------------------------------- #

def chrom_index(player_val: int, dealer_val: int, is_soft: bool):
    """Return bit position in chromosome (bits 0-259) for the given state.

    Returns None when the state maps to always-stand (player_val == 21).
    """
    if player_val == 21:
        return None

    dealer_idx = _DEALER_IDX.get(dealer_val)
    if dealer_idx is None:
        return None

    if is_soft:
        # Soft 12–20 → rows 0–8, offset 170
        if not (12 <= player_val <= 20):
            return None
        return 170 + (player_val - 12) * 10 + dealer_idx
    else:
        # Hard 4–20 → rows 0–16
        if not (4 <= player_val <= 20):
            return None
        return (player_val - 4) * 10 + dealer_idx


def decode_count_values(chrom: np.ndarray) -> list[int]:
    """Decode bits 260–281 into 11 count values.

    Each 2-bit group: 00 → -1, 01 → 0, 10 → +1, 11 → 0.
    """
    result = []
    for i in range(11):
        b0 = int(chrom[260 + i * 2])
        b1 = int(chrom[260 + i * 2 + 1])
        val = b0 * 2 + b1
        if val == 2:       # 10
            result.append(+1)
        elif val == 0:     # 00
            result.append(-1)
        else:              # 01 or 11
            result.append(0)
    return result


def decode_bet_multipliers(chrom: np.ndarray) -> list[int]:
    """Decode bits 282–293 into 4 bet multipliers (1–8).

    Each 3-bit group encodes value 0–7; multiplier = value + 1.
    """
    result = []
    for i in range(4):
        b0 = int(chrom[282 + i * 3])
        b1 = int(chrom[282 + i * 3 + 1])
        b2 = int(chrom[282 + i * 3 + 2])
        val = b0 * 4 + b1 * 2 + b2
        result.append(val + 1)
    return result


def make_agent(chrom: np.ndarray):
    """Wrap chromosome bits 0–259 into an agent callable: state → action."""
    def agent(state):
        player_val, dealer_val, is_soft_int = state
        is_soft = bool(is_soft_int)
        idx = chrom_index(player_val, dealer_val, is_soft)
        if idx is None:
            return 0  # stand on 21 or unknown states
        return int(chrom[idx])
    return agent


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #

def evaluate_fitness(chrom: np.ndarray, n_hands: int = N_HANDS) -> float:
    """Simulate n_hands and return a fitness score.

    Fitness = n_hands + final_bankroll when the session completes (survivor bonus
    guarantees every survivor outscores every bankrupt individual).
    Fitness = hands_survived when bankrupt (partial credit keeps selection
    pressure meaningful even when the whole population goes bust early).
    """
    cv = decode_count_values(chrom)
    bm = decode_bet_multipliers(chrom)
    env = CardCountingEnv(cv, bm)
    agent = make_agent(chrom)

    for hand_num in range(n_hands):
        state, info = env.reset()
        if env.bankroll <= 0:
            return float(hand_num)          # partial credit: hands survived
        if info.get("hand_done"):
            continue
        done = False
        while not done:
            action = agent(state)
            state, reward, done, info = env.step(action)
        if env.bankroll <= 0:
            return float(hand_num + 1)      # partial credit: include this hand

    # Survivor bonus: guaranteed > any bankrupt score (max partial = n_hands)
    return float(n_hands) + env.bankroll


def simulate_bankroll_trajectory(chrom: np.ndarray, n_hands: int = N_HANDS) -> list[float]:
    """Run n_hands and record bankroll after each completed hand."""
    cv = decode_count_values(chrom)
    bm = decode_bet_multipliers(chrom)
    env = CardCountingEnv(cv, bm)
    agent = make_agent(chrom)
    trajectory = [env.bankroll]

    for _ in range(n_hands):
        state, info = env.reset()
        if env.bankroll <= 0:
            trajectory.append(0.0)
            break
        if not info.get("hand_done"):
            done = False
            while not done:
                action = agent(state)
                state, reward, done, info = env.step(action)
        trajectory.append(env.bankroll)
        if env.bankroll <= 0:
            break

    return trajectory


# --------------------------------------------------------------------------- #
# GA operators
# --------------------------------------------------------------------------- #

def init_population() -> np.ndarray:
    """Random binary population of shape (POP_SIZE, CHROM_LEN)."""
    return np.random.randint(0, 2, size=(POP_SIZE, CHROM_LEN), dtype=np.int8)


def roulette_select(population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
    """Fitness-proportionate selection with min-shift for robustness."""
    shifted = fitnesses - fitnesses.min()
    total = shifted.sum()
    if total == 0:
        probs = np.ones(len(fitnesses)) / len(fitnesses)
    else:
        probs = shifted / total
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

    history is a list of (min, max, median, mean) bankroll per generation.
    """
    population = init_population()
    history = []

    for gen in range(N_GENERATIONS):
        fitnesses = np.array([evaluate_fitness(ind) for ind in population])

        stats = (
            float(fitnesses.min()),
            float(fitnesses.max()),
            float(np.median(fitnesses)),
            float(fitnesses.mean()),
        )
        history.append(stats)
        print(
            f"Gen {gen + 1:3d}/{N_GENERATIONS}  "
            f"min=${stats[0]:.0f}  max=${stats[1]:.0f}  "
            f"med=${stats[2]:.0f}  mean=${stats[3]:.0f}"
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

        # Fill via roulette selection → crossover → mutation
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
# Output / visualisation
# --------------------------------------------------------------------------- #

def plot_fitness(history: list, save_dir: str = "."):
    """Line plot of min/max/median/mean fitness vs generation.

    Fitness = hands_survived when bankrupt, or N_HANDS + bankroll when surviving.
    The dashed line at N_HANDS marks the survivor threshold.
    """
    gens = range(1, len(history) + 1)
    mins, maxs, meds, means = zip(*history)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(N_HANDS, color="black", linewidth=1.2, linestyle="--",
               label=f"Survivor threshold ({N_HANDS} hands)", alpha=0.6)
    ax.axhline(N_HANDS + 1000, color="gray", linewidth=1, linestyle=":",
               label="$1,000 bankroll line", alpha=0.5)
    ax.plot(gens, maxs,  label="Max",    linewidth=2)
    ax.plot(gens, means, label="Mean",   linewidth=2, linestyle="--")
    ax.plot(gens, meds,  label="Median", linewidth=2, linestyle="-.")
    ax.plot(gens, mins,  label="Min",    linewidth=2, linestyle=":")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (hands survived  or  N_HANDS + bankroll)")
    ax.set_title("GA Convergence — Card Counting Strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "fitness_history.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_strategy_heatmap(population: np.ndarray, save_dir: str = "."):
    """Two side-by-side heatmaps: fraction of population recommending Hit.

    Hard hands (rows=player totals 4-20, cols=dealer 2-A).
    Soft hands (rows=player totals soft 12-20, cols=dealer 2-A).
    """
    dealer_labels = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
    hard_rows = list(range(4, 21))   # 17 values
    soft_rows = list(range(12, 21))  # 9 values

    hard_matrix = np.zeros((len(hard_rows), 10))
    for r, pv in enumerate(hard_rows):
        for c, dv in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
            idx = chrom_index(pv, dv, is_soft=False)
            if idx is not None:
                hard_matrix[r, c] = population[:, idx].mean()

    soft_matrix = np.zeros((len(soft_rows), 10))
    for r, pv in enumerate(soft_rows):
        for c, dv in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
            idx = chrom_index(pv, dv, is_soft=True)
            if idx is not None:
                soft_matrix[r, c] = population[:, idx].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, matrix, row_labels, title in [
        (axes[0], hard_matrix, [str(v) for v in hard_rows], "Hard Hands"),
        (axes[1], soft_matrix, [f"A+{v - 11}" for v in soft_rows], "Soft Hands"),
    ]:
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(10))
        ax.set_xticklabels(dealer_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Dealer Upcard")
        ax.set_ylabel("Player Total")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="% Recommending Hit")

    fig.suptitle("Evolved Card-Counting Strategy (Blue=Stand, Red=Hit)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "strategy_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_bankroll_trajectory(chrom: np.ndarray, save_dir: str = "."):
    """Bankroll over 1000 hands for the best individual."""
    trajectory = simulate_bankroll_trajectory(chrom)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(trajectory)), trajectory, linewidth=1.2, color="steelblue")
    ax.axhline(1000, color="black", linewidth=1, linestyle="--", alpha=0.5, label="Starting bankroll ($1,000)")
    ax.set_xlabel("Hand")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Best Individual — Bankroll Trajectory (1000 Hands)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "bankroll_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def print_count_table(chrom: np.ndarray):
    """Print evolved count values vs Hi-Lo reference."""
    cv = decode_count_values(chrom)
    print("\n=== Evolved Count Values vs Hi-Lo ===")
    print(f"{'Card Group':<18} {'Hi-Lo':>8} {'Evolved':>8}")
    print("-" * 38)
    for name, hilo, evolved in zip(_GROUP_NAMES, _HILO_REF, cv):
        marker = " ✓" if evolved == hilo else "  "
        print(f"{name:<18} {hilo:>+8} {evolved:>+8}{marker}")


def print_bet_table(chrom: np.ndarray):
    """Print bet multiplier and dollar bet per true-count range."""
    bm = decode_bet_multipliers(chrom)
    print("\n=== Bet Sizing by True Count ===")
    print(f"{'True Count Range':<18} {'Multiplier':>12} {'Dollar Bet':>12}")
    print("-" * 44)
    for label, mult in zip(_TC_RANGE_LABELS, bm):
        dollar = mult * BASE_BET
        print(f"{label:<18} {mult:>12}x   ${dollar:>9}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    save_dir = os.path.dirname(os.path.abspath(__file__))

    print(
        f"Starting GA: pop={POP_SIZE}, generations={N_GENERATIONS}, "
        f"hands/eval={N_HANDS}, mutation={MUTATION_RATE}/bit\n"
    )

    final_pop, history = run_ga()

    # Re-evaluate final population to find best chromosome
    print("\nRe-evaluating final population to select best individual...")
    final_fitnesses = np.array([evaluate_fitness(ind) for ind in final_pop])
    best_idx = int(np.argmax(final_fitnesses))
    best_chrom = final_pop[best_idx]
    best_score = final_fitnesses[best_idx]
    best_bankroll = best_score - N_HANDS if best_score > N_HANDS else 0.0
    print(f"Best fitness score: {best_score:.0f}  (bankroll: ${best_bankroll:.2f})")

    # Text output
    print_count_table(best_chrom)
    print_bet_table(best_chrom)

    # Plots
    print("\nGenerating plots...")
    plot_fitness(history, save_dir=save_dir)
    plot_strategy_heatmap(final_pop, save_dir=save_dir)
    plot_bankroll_trajectory(best_chrom, save_dir=save_dir)

    print("\nDone.")
