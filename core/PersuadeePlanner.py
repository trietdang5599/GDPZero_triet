import random
from typing import List

from core.game import PersuasionGame


class PersuadeeHeuristicPlanner:
	def __init__(self, dialog_acts: List[str], donate_prob: float = 0.4, seed: int | None = None):
		self.dialog_acts = dialog_acts
		self.donate_prob = max(0.0, min(1.0, donate_prob))
		self.rng = random.Random(seed)

	def _weighted_choice(self, candidates: List[str]) -> str:
		available = [da for da in candidates if da in self.dialog_acts]
		if not available:
			return PersuasionGame.U_Neutral if PersuasionGame.U_Neutral in self.dialog_acts else self.dialog_acts[0]
		return self.rng.choice(available)

	def select_action(self, state) -> str:
		if len(state) == 0:
			return self._weighted_choice([PersuasionGame.U_Neutral])
		last_role, last_da, _ = state[-1]
		if last_role != PersuasionGame.SYS:
			return self._weighted_choice([PersuasionGame.U_Neutral])

		mapping = {
			PersuasionGame.S_Greeting: [PersuasionGame.U_PositiveReaction, PersuasionGame.U_Neutral],
			PersuasionGame.S_CredibilityAppeal: [PersuasionGame.U_PositiveReaction, PersuasionGame.U_Neutral, PersuasionGame.U_NegativeReaction],
			PersuasionGame.S_EmotionAppeal: [PersuasionGame.U_PositiveReaction, PersuasionGame.U_Neutral, PersuasionGame.U_NegativeReaction],
			PersuasionGame.S_TaskRelatedInquiry: [PersuasionGame.U_PositiveReaction, PersuasionGame.U_Neutral],
			PersuasionGame.S_LogicalAppeal: [PersuasionGame.U_PositiveReaction, PersuasionGame.U_Neutral],
			PersuasionGame.S_Other: [PersuasionGame.U_Neutral, PersuasionGame.U_PositiveReaction],
		}
		if last_da == PersuasionGame.S_PropositionOfDonation:
			if self.rng.random() < self.donate_prob and PersuasionGame.U_Donate in self.dialog_acts:
				return PersuasionGame.U_Donate
			return self._weighted_choice([
				PersuasionGame.U_PositiveReaction,
				PersuasionGame.U_NoDonation,
				PersuasionGame.U_Neutral,
			])
		return self._weighted_choice(mapping.get(last_da, [PersuasionGame.U_Neutral]))
