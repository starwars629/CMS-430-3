# Blackjack GA Solver

A genetic algorithm that evolves optimal blackjack strategies, plus a simple simulation environment for testing agents.

## Project Structure

```
blackjack/
├── card.py          # Card model (suits, ranks, values)
├── deck.py          # 52-card deck with shuffle and deal
├── hand.py          # Hand model (value, soft/bust detection)
├── env.py           # BlackjackEnv — step-based simulation environment
├── demo.py          # Random and threshold-17 baseline agents
└── ga_solver.py     # Genetic algorithm that evolves a strategy
```

## Environment

`BlackjackEnv` follows a simple step-based interface:

```python
env = BlackjackEnv()
state, info = env.reset()       # (player_total, dealer_upcard, is_soft)
state, reward, done, info = env.step(action)  # action: 0=Stand, 1=Hit
```

Rewards: `1.0` = win, `0.5` = push, `0.0` = loss/bust.

## Running the Demos

```bash
cd blackjack
python demo.py
```

Runs 10,000 episodes each for a random agent and a threshold-17 agent and prints their win rates.

## Running the Genetic Algorithm

```bash
cd blackjack
python ga_solver.py
```

Runs 100 generations of 100 individuals, each evaluated over 1,000 hands. Prints per-generation stats and saves two output files.

### Output Files

| File | Description |
|---|---|
| `fitness_history.png` | Convergence plot — min/max/median/mean fitness per generation |
| `strategy_heatmap.png` | Evolved strategy heatmaps for hard and soft hands |

## Chromosome Design

Each individual is a 260-bit binary string:

| Segment    | Bits    | Player Totals       | Dealer Upcards    |
|------------|---------|---------------------|-------------------|
| Hard hands | 0–169   | 4–20 (17 rows)      | 2–10, A (10 cols) |
| Soft hands | 170–259 | Soft 12–20 (9 rows) | 2–10, A (10 cols) |

`0` = Stand, `1` = Hit. Player total 21 always stands (no lookup).

## GA Parameters

| Parameter               | Value                                  |
|-------------------------|----------------------------------------|
| Population size         | 100                                    |
| Generations             | 100                                    |
| Episodes per evaluation | 1,000                                  |
| Mutation rate           | 0.01 per bit                           |
| Elitism                 | Top 2 carried unchanged                |
| Selection               | Roulette wheel (fitness-proportionate) |
| Crossover               | Single-point                           |

## Dependencies

- Python 3.10+
- `numpy`
- `matplotlib`

Install with:

```bash
pip install numpy matplotlib
```
