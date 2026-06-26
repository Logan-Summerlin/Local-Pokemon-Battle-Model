"""Canonical action vocabulary and encoding for Pokemon battles.

Defines the fixed action space for Gen 3 OU singles:
- 4 move slots = 4 move actions
- Up to 5 switch targets = 5 switch actions
- Total: 9 canonical action indices (not all legal at every turn)

The action space is intentionally over-complete — legal action masks
restrict which actions are available on each turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActionType(IntEnum):
    """Types of battle actions."""

    MOVE = 0
    SWITCH = 1


# Canonical action indices
# Moves: 0-3 (move 1-4)
# Switches: 4-8 (switch to team slot 2-6, since slot 1 is active)
MOVE_1 = 0
MOVE_2 = 1
MOVE_3 = 2
MOVE_4 = 3
SWITCH_2 = 4
SWITCH_3 = 5
SWITCH_4 = 6
SWITCH_5 = 7
SWITCH_6 = 8

NUM_ACTIONS = 9  # Total size of action vocabulary

# Action names for logging
ACTION_NAMES = [
    "move 1",
    "move 2",
    "move 3",
    "move 4",
    "switch 2",
    "switch 3",
    "switch 4",
    "switch 5",
    "switch 6",
]


@dataclass(frozen=True, slots=True)
class BattleAction:
    """A specific action to take in battle."""

    action_type: ActionType
    move_index: int = -1  # 0-3 for moves, -1 for switches
    switch_index: int = -1  # 0-4 (team position, 0-indexed from non-active) for switches

    @property
    def canonical_index(self) -> int:
        """Get the index in the canonical action vocabulary."""
        if self.action_type == ActionType.SWITCH:
            # switch_index is 0-indexed position in team (excluding active)
            # maps to indices 4-8
            return SWITCH_2 + self.switch_index
        else:
            return MOVE_1 + self.move_index

    def to_showdown_command(self, move_names: list[str] | None = None) -> str:
        """Convert to a Showdown protocol command string.

        Args:
            move_names: Optional list of move names (for logging; not used in protocol).
                       The protocol uses 1-indexed move numbers.
        """
        if self.action_type == ActionType.SWITCH:
            # Showdown uses 1-indexed team positions
            # switch_index 0 = team slot 2 (1-indexed), etc.
            return f"/choose switch {self.switch_index + 2}"
        else:
            move_num = self.move_index + 1  # 1-indexed
            return f"/choose move {move_num}"

    def __repr__(self) -> str:
        if self.canonical_index < len(ACTION_NAMES):
            return f"BattleAction({ACTION_NAMES[self.canonical_index]})"
        return f"BattleAction(index={self.canonical_index})"


def action_from_canonical_index(index: int) -> BattleAction:
    """Create a BattleAction from a canonical action index."""
    if index < 0 or index >= NUM_ACTIONS:
        raise ValueError(f"Invalid action index: {index}")

    if index <= MOVE_4:
        return BattleAction(action_type=ActionType.MOVE, move_index=index)
    else:
        return BattleAction(
            action_type=ActionType.SWITCH,
            switch_index=index - SWITCH_2,
        )


def action_from_showdown_choice(choice: str) -> BattleAction | None:
    """Parse a Showdown choice string into a BattleAction.

    Examples:
        "move 1" -> BattleAction(MOVE, move_index=0)
        "switch 4" -> BattleAction(SWITCH, switch_index=2)
    """
    choice = choice.strip().lower()
    if choice.startswith("/choose "):
        choice = choice[8:]

    parts = choice.split()
    if not parts:
        return None

    if parts[0] == "move" and len(parts) >= 2:
        try:
            move_num = int(parts[1])
        except ValueError:
            return None
        move_index = move_num - 1  # Convert to 0-indexed
        if move_index < 0 or move_index > 3:
            return None
        return BattleAction(action_type=ActionType.MOVE, move_index=move_index)

    elif parts[0] == "switch" and len(parts) >= 2:
        try:
            team_pos = int(parts[1])
        except ValueError:
            return None
        # team_pos is 1-indexed in Showdown protocol
        # switch_index is offset from SWITCH_2 (0 = slot 2, 1 = slot 3, etc.)
        switch_index = team_pos - 2
        if switch_index < 0 or switch_index > 4:
            return None
        return BattleAction(action_type=ActionType.SWITCH, switch_index=switch_index)

    return None


class ActionMask:
    """Binary mask over the canonical action vocabulary.

    A value of True at index i means action i is legal.
    """

    def __init__(self) -> None:
        self._mask = [False] * NUM_ACTIONS

    @classmethod
    def all_moves(cls) -> ActionMask:
        """Create a mask with all moves legal."""
        mask = cls()
        for i in range(MOVE_1, MOVE_4 + 1):
            mask._mask[i] = True
        return mask

    @classmethod
    def from_list(cls, legal_indices: list[int]) -> ActionMask:
        """Create a mask from a list of legal action indices."""
        mask = cls()
        for idx in legal_indices:
            if 0 <= idx < NUM_ACTIONS:
                mask._mask[idx] = True
        return mask

    def set_legal(self, index: int) -> None:
        """Mark an action as legal."""
        if 0 <= index < NUM_ACTIONS:
            self._mask[index] = True

    def set_illegal(self, index: int) -> None:
        """Mark an action as illegal."""
        if 0 <= index < NUM_ACTIONS:
            self._mask[index] = False

    def is_legal(self, index: int) -> bool:
        """Check if an action is legal."""
        if 0 <= index < NUM_ACTIONS:
            return self._mask[index]
        return False

    @property
    def legal_indices(self) -> list[int]:
        """Get list of legal action indices."""
        return [i for i, legal in enumerate(self._mask) if legal]

    @property
    def legal_actions(self) -> list[BattleAction]:
        """Get list of legal BattleActions."""
        return [action_from_canonical_index(i) for i in self.legal_indices]

    @property
    def num_legal(self) -> int:
        """Count of legal actions."""
        return sum(self._mask)

    @property
    def any_legal(self) -> bool:
        """Whether any action is legal."""
        return any(self._mask)

    def to_list(self) -> list[bool]:
        """Get the mask as a list of booleans."""
        return list(self._mask)

    def to_int_list(self) -> list[int]:
        """Get the mask as a list of 0/1 integers (for tensorization)."""
        return [int(x) for x in self._mask]

    def __repr__(self) -> str:
        legal = [ACTION_NAMES[i] for i in self.legal_indices]
        return f"ActionMask({legal})"
