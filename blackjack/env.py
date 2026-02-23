from deck import Deck
from hand import Hand


def _play_dealer(hand: Hand, deck: Deck) -> None:
    while hand.value() < 17:
        hand.add_card(deck.deal())


def _determine_outcome(player: Hand, dealer: Hand) -> float:
    if player.is_bust():
        return 0.0
    if dealer.is_bust():
        return 1.0
    if player.value() > dealer.value():
        return 1.0
    if player.value() < dealer.value():
        return 0.0
    return 0.5  # push


class BlackjackEnv:
    def __init__(self) -> None:
        self._deck: Deck | None = None
        self._player_hand: Hand | None = None
        self._dealer_hand: Hand | None = None
        self._done: bool = True

    def reset(self) -> tuple[tuple[int, int, int], dict]:
        self._deck = Deck()
        self._player_hand = Hand()
        self._dealer_hand = Hand()
        self._done = False

        # Interleaved deal: player, dealer, player, dealer
        self._player_hand.add_card(self._deck.deal())
        self._dealer_hand.add_card(self._deck.deal())
        self._player_hand.add_card(self._deck.deal())
        self._dealer_hand.add_card(self._deck.deal())

        state = self._get_state()
        info = {"player_hand": self._player_hand.cards(), "dealer_hand": self._dealer_hand.cards()}
        return state, info

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action not in (0, 1):
            raise ValueError(f"Invalid action: {action}. Must be 0 (stand) or 1 (hit).")

        if action == 1:  # HIT
            self._player_hand.add_card(self._deck.deal())
            if self._player_hand.is_bust():
                self._done = True
                return self._get_state(), 0.0, True, {"result": "bust"}
            return self._get_state(), 0.0, False, {}

        else:  # STAND (action == 0)
            _play_dealer(self._dealer_hand, self._deck)
            reward = _determine_outcome(self._player_hand, self._dealer_hand)
            self._done = True
            info = {
                "player_value": self._player_hand.value(),
                "dealer_value": self._dealer_hand.value(),
                "result": "win" if reward == 1.0 else "loss" if reward == 0.0 else "push",
            }
            return self._get_state(), reward, True, info

    def _get_state(self) -> tuple[int, int, int]:
        player_value = self._player_hand.value()
        dealer_upcard_value = self._dealer_hand.cards()[0].value()
        is_soft = int(self._player_hand.is_soft())
        return (player_value, dealer_upcard_value, is_soft)
