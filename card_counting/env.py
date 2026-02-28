from shoe import Shoe
from hand import Hand
from card import Card

BASE_BET = 10          # dollars per unit; multipliers 1-8 → bets $10–$80
STARTING_BANKROLL = 1000.0

# True-count range boundaries → bet_multipliers index
# Index 0: <= -2
# Index 1: -1 to +1
# Index 2: +2 to +4
# Index 3: >= +5


class CardCountingEnv:
    """Stateful blackjack environment with 6-deck shoe, bankroll, and Hi-Lo-style count.

    Args:
        count_values:    11 integers (one per card group, Ace…Face) each in {-1, 0, +1}.
        bet_multipliers: 4 integers (one per true-count range) each in 1–8.
    """

    def __init__(self, count_values: list[int], bet_multipliers: list[int]) -> None:
        self._count_values = count_values
        self._bet_multipliers = bet_multipliers
        self._shoe = Shoe()
        self._bankroll: float = STARTING_BANKROLL
        self._running_count: int = 0
        self._player_hand: Hand | None = None
        self._dealer_hand: Hand | None = None
        self._hole_card: Card | None = None
        self._current_bet: float = BASE_BET

    # ---------------------------------------------------------------------- #
    # Public properties
    # ---------------------------------------------------------------------- #

    @property
    def bankroll(self) -> float:
        return self._bankroll

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _true_count(self) -> int:
        decks = self._shoe.decks_remaining()
        if decks <= 0:
            return 0
        return round(self._running_count / decks)

    def _bet_range_index(self, true_count: int) -> int:
        if true_count <= -2:
            return 0
        elif true_count <= 1:
            return 1
        elif true_count <= 4:
            return 2
        else:
            return 3

    def _count_card(self, card: Card) -> None:
        self._running_count += self._count_values[card.count_group()]

    def _get_state(self) -> tuple[int, int, int]:
        pv = self._player_hand.value()
        du = self._dealer_hand.cards()[0].value()
        is_soft = int(self._player_hand.is_soft())
        return (pv, du, is_soft)

    def _resolve_stand(self) -> float:
        """Determine win/loss/push after dealer plays; update bankroll. Returns dollar P&L."""
        pv = self._player_hand.value()
        dv = self._dealer_hand.value()

        if self._dealer_hand.is_bust() or pv > dv:
            self._bankroll += self._current_bet
            return self._current_bet
        elif pv < dv:
            self._bankroll -= self._current_bet
            return -self._current_bet
        else:
            return 0.0  # push

    # ---------------------------------------------------------------------- #
    # Episode interface
    # ---------------------------------------------------------------------- #

    def reset(self) -> tuple[tuple[int, int, int], dict]:
        """Start a new hand. Reshuffles shoe if penetration threshold reached.

        Returns:
            state: (player_value, dealer_upcard_value, is_soft)
            info:  {"hand_done": bool, ...}

        When info["hand_done"] is True the hand resolved inside reset() (natural
        blackjack scenario) and no call to step() is needed.
        """
        # Reshuffle check
        if self._shoe.needs_reshuffle():
            self._shoe.reshuffle()
            self._running_count = 0

        # Bet sizing based on current true count
        tc = self._true_count()
        idx = self._bet_range_index(tc)
        multiplier = self._bet_multipliers[idx]
        self._current_bet = min(multiplier * BASE_BET, self._bankroll)

        # Deal interleaved: p1, d1(upcard), p2, d2(hole)
        self._player_hand = Hand()
        self._dealer_hand = Hand()

        p1 = self._shoe.deal()
        d1 = self._shoe.deal()
        p2 = self._shoe.deal()
        d2 = self._shoe.deal()

        self._player_hand.add_card(p1)
        self._dealer_hand.add_card(d1)
        self._player_hand.add_card(p2)
        self._dealer_hand.add_card(d2)

        # Count visible cards; hole card (d2) stays face-down
        self._count_card(p1)
        self._count_card(d1)
        self._count_card(p2)
        self._hole_card = d2

        state = self._get_state()

        # Natural blackjack check
        if self._player_hand.is_blackjack():
            self._count_card(d2)  # reveal hole card
            if self._dealer_hand.is_blackjack():
                # Push — no bankroll change
                return state, {"hand_done": True, "reward": 0.0, "result": "push"}
            else:
                # Player blackjack pays 3:2
                payout = 1.5 * self._current_bet
                self._bankroll += payout
                return state, {"hand_done": True, "reward": payout, "result": "blackjack"}

        return state, {"hand_done": False}

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool, dict]:
        """Take a hit (1) or stand (0).

        Returns:
            state:  current game state
            reward: dollar P&L for this action (non-zero only on hand completion)
            done:   True when the hand is complete
            info:   diagnostic dict
        """
        if action == 1:  # Hit
            card = self._shoe.deal()
            self._player_hand.add_card(card)
            self._count_card(card)
            if self._player_hand.is_bust():
                self._bankroll -= self._current_bet
                return self._get_state(), -self._current_bet, True, {"result": "bust"}
            return self._get_state(), 0.0, False, {}

        else:  # Stand (action == 0)
            # Reveal and count hole card
            self._count_card(self._hole_card)

            # Dealer plays: stand on soft 17+
            while self._dealer_hand.value() < 17:
                card = self._shoe.deal()
                self._dealer_hand.add_card(card)
                self._count_card(card)

            reward = self._resolve_stand()
            return self._get_state(), reward, True, {"result": "done"}
