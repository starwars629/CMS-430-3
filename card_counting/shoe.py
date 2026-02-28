import random
from card import Card, SUITS, RANKS

N_DECKS = 6
TOTAL_CARDS = N_DECKS * 52        # 312
RESHUFFLE_THRESHOLD = 78          # <= 78 remaining (25%) triggers reshuffle


class Shoe:
    def __init__(self) -> None:
        self._cards: list[Card] = []
        self.reshuffle()

    def reshuffle(self) -> None:
        """Rebuild all 312 cards and shuffle."""
        self._cards = [Card(suit, rank) for _ in range(N_DECKS) for suit in SUITS for rank in RANKS]
        random.shuffle(self._cards)

    def deal(self) -> Card:
        if not self._cards:
            raise RuntimeError("Shoe is empty")
        return self._cards.pop()

    def cards_remaining(self) -> int:
        return len(self._cards)

    def decks_remaining(self) -> float:
        return self.cards_remaining() / 52

    def needs_reshuffle(self) -> bool:
        """True when 75% penetration reached (≤78 cards left)."""
        return self.cards_remaining() <= RESHUFFLE_THRESHOLD
