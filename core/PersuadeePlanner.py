import logging
import random
from typing import List, Optional

from core.game import PersuasionGame
from core.gen_models import GenerationModel
from core.helpers import DialogSession

logger = logging.getLogger(__name__)


class PersuadeeHeuristicPlanner:
	def __init__(
		self,
			dialog_acts: List[str],
			generation_model: GenerationModel,
			max_hist_num_turns: int = 2,
			donate_prob: float | None = None,
			seed: int | None = None,
	):
		if generation_model is None:
			raise ValueError("generation_model is required for context-aware persuadee planning.")
		self.dialog_acts = dialog_acts
		if donate_prob is not None:
			logger.debug("donate_prob is deprecated and ignored. Persuadee now uses LLM-based planning.")
		self._llm_planner = PersuadeeLLMPlanner(
			dialog_acts=dialog_acts,
			generation_model=generation_model,
			max_hist_num_turns=max_hist_num_turns,
			seed=seed,
		)

	def select_action(self, state: DialogSession) -> str:
		return self._llm_planner.select_action(state)


class PersuadeeLLMPlanner:
	"""
	Content-aware persuadee planner that infers the next user dialog act
	from the latest 1–2 Persuader utterances using an LLM classifier-style prompt.

	It returns only a dialog act (no text). Downstream NLG should render the utterance.
	"""

	def __init__(
		self,
			dialog_acts: List[str],
			generation_model: GenerationModel,
			max_hist_num_turns: int = 2,
			seed: Optional[int] = None,
	):
		self.dialog_acts = dialog_acts
		self.model = generation_model
		self.max_hist_num_turns = max(1, int(max_hist_num_turns))
		self.rng = random.Random(seed)
		# Deterministic, classifier-like decoding
		self.classifier_args = {
			"max_new_tokens": 16,
			"temperature": 0.0,
			"do_sample": False,
			"return_full_text": False,
		}

	def _normalize_da(self, candidate: str) -> Optional[str]:
		cand = (candidate or "").strip().lower()
		# Strip surrounding brackets if present
		if cand.startswith("[") and "]" in cand:
			cand = cand[1: cand.index("]")].strip()
		for da in self.dialog_acts:
			if cand == da.lower():
				return da
		return None

	def _fallback(self, state: DialogSession) -> str:
		# Simple fallback mirroring heuristic neutrality
		return (
			PersuasionGame.U_Neutral
			if PersuasionGame.U_Neutral in self.dialog_acts
			else self.dialog_acts[0]
		)

	def _build_prompt(self, state: DialogSession) -> str:
		# Collect last 1–2 Persuader utterances from the state
		sys_utts: List[str] = []
		for role, _da, utt in reversed(state):
			if role == PersuasionGame.SYS:
				sys_utts.append(str(utt).strip())
				if len(sys_utts) >= self.max_hist_num_turns:
					break
		# Present most-recent first in the prompt for clarity
		sys_utts = list(reversed(sys_utts))
		acts = " ".join([f"[{da}]" for da in self.dialog_acts])
		context = "\n".join([f"Persuader: {u}" for u in sys_utts])
		instruction = (
			"You are selecting the Persuadee's dialog act for the next turn.\n"
			"Choose exactly one from: " + acts + ".\n"
			"Answer with only the label in brackets, e.g., [donate]."
		)
		return (
			"The conversation context (most recent first if multiple):\n"
			+ context
			+ "\n\n"
			+ instruction
		)

	def select_action(self, state: DialogSession) -> str:
		# Expect that last role is Persuader; otherwise fall back
		if len(state) == 0 or state[-1][0] != PersuasionGame.SYS:
			return self._fallback(state)
		try:
			prompt = self._build_prompt(state)
			data = self.model.generate(prompt, **self.classifier_args)
			# Prefer cleaned response if available on the model
			resp = None
			try:
				resp = self.model._cleaned_resp(data, prompt)[0]
			except Exception:  # pragma: no cover - robust fallback
				resp = data[0].get("generated_text", "").strip() if data else ""
			norm = self._normalize_da(resp)
			if norm:
				return norm
			# Try to extract bracketed token if model returned text like "[donate] I will ..."
			start = resp.find("[")
			end = resp.find("]")
			if start != -1 and end != -1 and end > start + 1:
				br = resp[start + 1:end].strip()
				norm = self._normalize_da(br)
				if norm:
					return norm
			logger.debug("LLM planner could not normalize DA from: %s", resp)
		except Exception as exc:  # pragma: no cover - best effort
			logger.debug("LLM planner failed to select action: %s", exc)
		return self._fallback(state)
