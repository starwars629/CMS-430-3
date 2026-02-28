SUITS = ("Hearts", "Diamonds", "Clubs", "Spades")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

_RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10,
    "A": 11,
}

# Maps rank to Hi-Lo count group index (0–10)
# Order: Ace=0, 2=1, 3=2, 4=3, 5=4, 6=5, 7=6, 8=7, 9=8, 10=9, Face(J/Q/K)=10
_COUNT_GROUP = {
    "A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
    "7": 6, "8": 7, "9": 8, "10": 9, "J": 10, "Q": 10, "K": 10,
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

    def count_group(self) -> int:
        return _COUNT_GROUP[self._rank]

    def __repr__(self) -> str:
        return f"{self._rank} of {self._suit}"
