import logging
import random
from typing import List, Optional

import numpy as np


from core.helpers import DialogSession
from core.gen_models import GenerationModel
from core.game import PersuasionGame

logger = logging.getLogger(__name__)


class PersuaderLLMPlanner:
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
		if len(state) == 0:
			context = "Persuadee: Hello!"
		else:
			sys_utts: List[str] = []
			for role, _da, utt in reversed(state):
				if role == PersuasionGame.USR:
					sys_utts.append(f"Persuadee: {utt}")
					if len(sys_utts) >= self.max_hist_num_turns:
						break
			if not sys_utts:
				sys_utts = ["Persuadee: Hello!"]
			context = "\n".join(reversed(sys_utts))

		act_definitions = "\n".join(
			f"[{da}] Describe briefly how this act advances persuasion."
			for da in self.dialog_acts
		)
		options = " ".join(f"[{da}]" for da in self.dialog_acts)
		guidelines = "\n".join(
			[
				"- Pick the single dialog act that best advances the persuasion objective.",
				"- Respond with only one label in brackets, e.g., [credibility appeal].",
			]
		)
		return "\n\n".join(
			[
				"The conversation context:",
				context,
				"Dialog act options:",
				act_definitions,
				"Guidelines:",
				guidelines,
				f"Available labels: {options}",
				"Chosen label:",
			]
		)

	def select_action(self, state: DialogSession) -> str:
		# Expect that the buyer has just spoken; otherwise fall back.
		valid_mask = self.get_valid_moves(state)
		if len(state) == 0 or state[-1][0] != PersuasionGame.USR:
			return self.dialog_acts[self._fallback_index(valid_mask)]

		try:
			prompt = self._build_prompt(state)
			data = self.model.generate(prompt, **self.selection_args)
			resp = None
			try:
				resp = self.model._cleaned_resp(data, prompt)[0]
			except Exception:
				resp = data[0].get("generated_text", "").strip() if data else ""
			normalized = self._normalize_da(resp)
			if not normalized and resp:
				start = resp.find("[")
				end = resp.find("]")
				if start != -1 and end != -1 and end > start + 1:
					br = resp[start + 1 : end].strip()
					normalized = self._normalize_da(br)
			if normalized and normalized in self.dialog_acts:
				idx = self.dialog_acts.index(normalized)
				if valid_mask[idx]:
					return normalized
			logger.debug("PersuaderLLMPlanner could not normalize DA from: %s", resp)
		except Exception as exc:  # pragma: no cover
			logger.debug("PersuaderLLMPlanner failed to select action: %s", exc)

		return self.dialog_acts[self._fallback_index(valid_mask)]


__all__ = ["PersuaderLLMPlanner"]
