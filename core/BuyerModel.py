import logging
import re
from typing import List, Optional, Tuple

from core.game import NegotiationGame
from core.gen_models import DialogModel, GenerationModel
from core.helpers import DialogSession
from utils.dialog_acts import BUYER_DIALOG_ACT_DEFINITIONS
from utils.utils import log_prompt


logger = logging.getLogger(__name__)


RESP_PATTERN = re.compile(r"\s*\[([^\]]+)\]\s*(.*)", re.DOTALL)
TAG_PREFIX = re.compile(r"^\s*\[[^\]]+\]\s*")


class BuyerModel(DialogModel):
	"""LLM-backed buyer simulator for Craigslist Bargain negotiations."""

	def __init__(
		self,
		dialog_acts: List[str],
		inference_args: dict,
		backbone_model: GenerationModel,
		conv_examples: Optional[List[DialogSession]] = None,
		max_hist_num_turns: int = 5,
	):
		super().__init__()
		self.dialog_acts = dialog_acts
		self.backbone_model = backbone_model
		self.conv_examples = conv_examples or []
		self.max_hist_num_turns = max_hist_num_turns
		self.da_definitions = {
			da: BUYER_DIALOG_ACT_DEFINITIONS.get(da, "Respond politely.")
			for da in dialog_acts
		}
		dialog_act_list = " ".join(f"[{da}]" for da in self.dialog_acts)
		self.task_prompt = f"""
You are the Buyer negotiating over a Craigslist listing. Act like a real person who cares about price, item condition, and logistics.
Always respond in the format `[dialog_act] utterance`, where `dialog_act` is one of: {dialog_act_list}.
Keep utterances natural, reference prior turns when relevant, and avoid robotic phrasing.
Example negotiations:
{self._format_examples()}
--- End of examples ---
""".strip()

		default_args = {
			"max_new_tokens": 96,
			"temperature": 1.0,
			"repetition_penalty": 1.05,
			"do_sample": True,
			"return_full_text": False,
		}
		self.inference_args = {**default_args, **(inference_args or {})}

	def _format_examples(self) -> str:
		if not self.conv_examples:
			return "Seller: [seller-intro] Hi there!\nBuyer: [buyer-greeting] Hello!"
		return "\n\n".join(
			exp.to_string_rep(keep_sys_da=True, keep_user_da=True)
			for exp in self.conv_examples
		)

	def _build_prompt(self, state: DialogSession, forced_act: Optional[str] = None) -> str:
		history = state.to_string_rep(
			keep_sys_da=True,
			keep_user_da=True,
			max_turn_to_display=self.max_hist_num_turns,
		)
		act_instruction = ""
		if forced_act and forced_act in self.dialog_acts:
			definition = self.da_definitions.get(forced_act, "")
			act_instruction = (
				f"You must respond with dialog act [{forced_act}]. {definition} "
				"Keep the rest of the utterance natural."
			).strip()
		parts = [
			self.task_prompt,
			"Conversation so far:",
			history or "Seller: [seller-intro] Hi there!\nBuyer: [buyer-greeting] Hey!",
			act_instruction,
			"Buyer:",
		]
		return "\n\n".join(part for part in parts if part).strip()

	def _strip_tags(self, text: str) -> str:
		while True:
			match = TAG_PREFIX.match(text)
			if not match:
				break
			text = text[match.end() :].lstrip()
		return text

	def _parse_da_and_text(self, raw_text: str) -> Tuple[str, str]:
		match = RESP_PATTERN.match(raw_text or "")
		if not match:
			return NegotiationGame.B_OTHER, self._strip_tags(raw_text.strip()) or "I'm still thinking."
		da = match.group(1).strip()
		utterance = match.group(2).strip()
		if da not in self.dialog_acts:
			da = NegotiationGame.B_OTHER
		return da, self._strip_tags(utterance) or "I'm still thinking."

	def _clean_response(self, data) -> Tuple[str, str]:
		for resp in data:
			text = (resp.get("generated_text") or "").strip()
			if not text:
				continue
			if text.lower().startswith(f"{NegotiationGame.USR.lower()}:"):
				text = text.split(":", 1)[1].strip()
			return self._parse_da_and_text(text)
		return NegotiationGame.B_OTHER, "Could you clarify the offer?"

	def get_utterance(self, state: DialogSession, action) -> str:
		raise NotImplementedError("BuyerModel should not be used as the system agent.")

	def get_utterance_batched(self, state: DialogSession, action: int, batch: int) -> List[str]:
		raise NotImplementedError("BuyerModel should not be used as the system agent.")

	def get_utterance_w_da(self, state: DialogSession, action=None, **_kwargs) -> Tuple[str, str]:
		prompt = self._build_prompt(state, forced_act=action)
		log_prompt(f"[BUYER_MODEL]\n{prompt}")
		data = self.backbone_model.generate(prompt, **self.inference_args)
		da, utt = self._clean_response(data)
		logger.debug("Buyer responded with da=%s text=%s", da, utt)
		return da, utt

	def get_utterance_w_da_from_batched_states(self, states: List[DialogSession], action=None):
		das: List[str] = []
		utts: List[str] = []
		for dialog_state in states:
			da, utt = self.get_utterance_w_da(dialog_state, action=action)
			das.append(da)
			utts.append(utt)
		return das, utts


__all__ = ["BuyerModel"]
