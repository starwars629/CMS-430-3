from card import Card


class Hand:
    def __init__(self) -> None:
        self._cards: list[Card] = []

    def add_card(self, card: Card) -> None:
        self._cards.append(card)

    def cards(self) -> list[Card]:
        return list(self._cards)

    def value(self) -> int:
        total = sum(c.value() for c in self._cards)
        aces = sum(1 for c in self._cards if c.rank == "A")
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    def is_soft(self) -> bool:
        total = sum(c.value() for c in self._cards)
        aces = sum(1 for c in self._cards if c.rank == "A")
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        # If there are still aces counted as 11, it's soft
        return aces > 0

    def is_bust(self) -> bool:
        return self.value() > 21

    def is_blackjack(self) -> bool:
        return len(self._cards) == 2 and self.value() == 21
