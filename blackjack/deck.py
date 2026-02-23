import random
from card import Card, SUITS, RANKS


class Deck:
    def __init__(self) -> None:
        self._cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        random.shuffle(self._cards)

    def deal(self) -> Card:
        if not self._cards:
            raise RuntimeError("Deck is empty")
        return self._cards.pop()

    def cards_remaining(self) -> int:
        return len(self._cards)
