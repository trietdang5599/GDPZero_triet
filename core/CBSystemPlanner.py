import logging
import re
from typing import List, Tuple

import numpy as np

from core.dialog_planner import DialogPlanner
from core.game import NegotiationGame
from core.gen_models import GenerationModel
from core.helpers import DialogSession


logger = logging.getLogger(__name__)
DA_PATTERN = re.compile(r"\[([^\]]+)\]")


class CBSystemPlanner(DialogPlanner):
	"""Planner that predicts the next seller dialog act for Craigslist Bargains using an LLM."""

	def __init__(
		self,
		dialog_acts: List[str],
		max_hist_num_turns: int,
		user_dialog_acts: List[str],
		user_max_hist_num_turns: int,
		generation_model: GenerationModel,
		conv_examples: List[DialogSession] | None = None,
	):
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns
		self.user_dialog_acts = user_dialog_acts
		self.user_max_hist_num_turns = user_max_hist_num_turns
		self.generation_model = generation_model
		self.conv_examples = conv_examples or []
		self.smoothing = 1.0

		self.task_prompt = (
			"You are planning the Seller's strategy for a Craigslist bargaining chat. "
			"Each action must be emitted as `[dialog_act] short justification` where "
			"`dialog_act` is one of the Seller dialog acts. Example conversations:\n"
			f"{self._format_examples()}\n"
			"--- End of examples ---"
		).strip()

		self.inf_args = {
			"max_new_tokens": 32,
			"temperature": 0.9,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 16,
		}

	def _format_examples(self) -> str:
		if not self.conv_examples:
			return "Seller: [seller-intro] Hi there!\nBuyer: [buyer-greeting] Hello!"
		return "\n\n".join(
			exp.to_string_rep(keep_sys_da=True, keep_user_da=True)
			for exp in self.conv_examples
		)

	def _build_prompt(self, state: DialogSession) -> str:
		history = state.to_string_rep(
			keep_sys_da=True,
			keep_user_da=True,
			max_turn_to_display=self.max_hist_num_turns,
		)
		parts = [
			self.task_prompt,
			"Conversation so far:",
			history or "Seller: [seller-intro] Hi there!\nBuyer: [buyer-greeting] Hello!",
			"Predict the Seller's next action as `[dialog_act] justification`.",
		]
		return "\n\n".join(part for part in parts if part).strip()

	def _extract_dialog_act(self, text: str) -> str | None:
		match = DA_PATTERN.search(text or "")
		if not match:
			return None
		da = match.group(1).strip()
		return da if da in self.dialog_acts else None

	def get_valid_moves(self, state: DialogSession) -> np.ndarray:
		if len(state) < 1:
			return np.array([
				1 if da in {NegotiationGame.S_INTRO, NegotiationGame.S_INIT_PRICE} else 0
				for da in self.dialog_acts
			])
		return np.ones(len(self.dialog_acts))

	def predict(self, state: DialogSession) -> Tuple[np.ndarray, float]:
		prompt = self._build_prompt(state)
		data = self.generation_model.generate(prompt, **self.inf_args)
		prob = np.zeros(len(self.dialog_acts))
		prob += self.smoothing
		for resp in data:
			da = self._extract_dialog_act(resp.get("generated_text", ""))
			if not da:
				continue
			prob[self.dialog_acts.index(da)] += 1
		if np.sum(prob) == 0:
			prob = np.ones(len(self.dialog_acts))
		prob /= prob.sum()
		value = self.heuristic(state)
		return prob, value

	def heuristic(self, state: DialogSession) -> float:
		score = 0.0
		for role, da, _ in state:
			if role == NegotiationGame.USR:
				if da == NegotiationGame.B_ACCEPT:
					return 1.0
				if da in {NegotiationGame.B_REJECT, NegotiationGame.B_QUIT}:
					return -1.0
				if da in {NegotiationGame.B_COUNTER, NegotiationGame.B_OFFER}:
					score = max(score, 0.2)
			else:
				if da == NegotiationGame.S_ACCEPT:
					score = max(score, 0.6)
				if da in {NegotiationGame.S_REJECT, NegotiationGame.S_QUIT}:
					score = min(score, -0.6)
		if len(state) >= self.max_hist_num_turns:
			score = min(score, -0.2)
		return float(score)


__all__ = ["CBSystemPlanner"]

