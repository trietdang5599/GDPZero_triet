import logging
import random
from typing import List, Optional

import numpy as np

from core.P4GSystemPlanner import DialogPlanner
from core.helpers import DialogSession
from core.gen_models import GenerationModel
from core.game import PersuasionGame

logger = logging.getLogger(__name__)


class PersuaderLLMPlanner(DialogPlanner):
	"""
	Simple LLM-backed planner for the Persuader.

	It mirrors :class:`PersuadeeLLMPlanner` by prompting a generation model to pick the
	next dialog act directly from the recent conversation context. The planner outputs
	a probability distribution over dialog acts (built from multiple LLM samples) so
	downstream code can sample stochastically.
	"""

	def __init__(
		self,
		dialog_acts: List[str],
		generation_model: GenerationModel,
		max_hist_num_turns: int = 2,
		seed: Optional[int] = None,
	) -> None:
		self.dialog_acts = list(dialog_acts)
		self.model = generation_model
		self.max_hist_num_turns = max(1, int(max_hist_num_turns))
		self.rng = random.Random(seed)
		self.selection_args = {
			"max_new_tokens": 12,
			"temperature": 0.9,
			"do_sample": True,
			"num_return_sequences": 8,
			"return_full_text": False,
		}

	def _normalize_da(self, candidate: str) -> Optional[str]:
		text = (candidate or "").strip()
		if not text:
			return None
		if text.startswith("[") and "]" in text:
			text = text[1 : text.index("]")]
		text = text.strip().lower()
		for da in self.dialog_acts:
			if text == da.lower():
				return da
		return None

	def _fallback_index(self, valid_mask: np.ndarray) -> int:
		valid_indices = np.flatnonzero(valid_mask)
		if valid_indices.size == 0:
			return 0
		return int(self.rng.choice(valid_indices))

	def _build_prompt(self, state: DialogSession) -> str:
		history = state.to_string_rep(
			keep_sys_da=False,
			keep_user_da=False,
			max_turn_to_display=self.max_hist_num_turns,
		)
		act_list = " ".join([f"[{da}]" for da in self.dialog_acts])
		instruction = (
			"You are the Persuader in a donation conversation.\n"
			"Pick the next dialog act that advances the persuasion.\n"
			f"Available dialog acts: {act_list}.\n"
			"Respond with one label in brackets, e.g., [credibility appeal]."
		)
		segments: List[str] = []
		if history:
			segments.append(f"Conversation so far:\n{history}")
		segments.append(instruction)
		body = "\n\n".join(segments)
		return f"{body}\n\nNext dialog act:"

	def get_valid_moves(self, state: DialogSession) -> np.ndarray:
		if len(state) < 1:
			return np.array(
				[1 if da == PersuasionGame.S_Greeting else 0 for da in self.dialog_acts],
				dtype=np.int32,
			)
		return np.ones(len(self.dialog_acts), dtype=np.int32)

	def predict(self, state: DialogSession) -> "tuple[np.ndarray, float]":
		valid_mask = self.get_valid_moves(state)
		prompt = self._build_prompt(state)

		counts = np.zeros(len(self.dialog_acts), dtype=np.float64)
		try:
			data = self.model.generate(prompt, **self.selection_args)
		except Exception as exc:  # pragma: no cover - fallback to heuristic choice
			logger.debug("PersuaderLLMPlanner generation failed: %s", exc)
			idx = self._fallback_index(valid_mask)
			counts[idx] = 1.0
			return counts, 0.0

		for item in data or []:
			resp = ""
			try:
				resp = self.model._cleaned_resp([item], prompt)[0]
			except Exception:
				resp = item.get("generated_text", "")
			da = self._normalize_da(resp)
			if da:
				idx = self.dialog_acts.index(da)
				if valid_mask[idx]:
					counts[idx] += 1.0

		if counts.sum() == 0.0:
			idx = self._fallback_index(valid_mask)
			counts[idx] = 1.0

		prob = counts / counts.sum()
		return prob, 0.0


__all__ = ["PersuaderLLMPlanner"]
