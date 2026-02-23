SUITS = ("Hearts", "Diamonds", "Clubs", "Spades")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

_RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10,
    "A": 11,
}


class Card:
    def __init__(self, suit: str, rank: str) -> None:
        if suit not in SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        if rank not in RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        self._suit = suit
        self._rank = rank

    @property
    def suit(self) -> str:
        return self._suit

    @property
    def rank(self) -> str:
        return self._rank

    def value(self) -> int:
        return _RANK_VALUES[self._rank]

    def __repr__(self) -> str:
        return f"{self._rank} of {self._suit}"
