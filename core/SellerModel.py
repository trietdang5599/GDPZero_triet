import logging
import re
from typing import List, Optional

from core.game import NegotiationGame
from core.gen_models import DialogModel, GenerationModel
from core.helpers import DialogSession
from utils.dialog_acts import SELLER_DIALOG_ACT_DEFINITIONS
from utils.utils import log_prompt


logger = logging.getLogger(__name__)

TAG_PREFIX = re.compile(r"^\s*\[[^\]]+\]\s*")


class SellerModel(DialogModel):
	"""LLM-backed seller agent for Craigslist Bargain negotiations."""

	def __init__(
		self,
		dialog_acts: List[str],
		backbone_model: GenerationModel,
		max_hist_num_turns: int = 5,
		conv_examples: Optional[List[DialogSession]] = None,
		inference_args: Optional[dict] = None,
		context_snippet: Optional[str] = None,
	):
		super().__init__()
		self.dialog_acts = dialog_acts
		self.backbone_model = backbone_model
		self.max_hist_num_turns = max_hist_num_turns
		self.conv_examples = conv_examples or []
		self.context_snippet = context_snippet or (
			"You are the Craigslist seller negotiating the price of a second-hand item. "
			"Stay polite, justify pricing decisions with concrete facts, and keep responses grounded in everyday language."
		)

		self.da_prompts_mapping = {
			da: SELLER_DIALOG_ACT_DEFINITIONS.get(da, "Respond naturally.")
			for da in dialog_acts
		}

		self.task_prompt = (
			"This is a negotiation between a Seller and a Buyer on Craigslist. "
			"The Seller wants a fair deal without sounding robotic. Example turns:\n"
			f"{self._format_examples()}\n"
			"--- End of examples ---"
		).strip()

		default_args = {
			"max_new_tokens": 96,
			"temperature": 0.7,
			"repetition_penalty": 1.05,
			"do_sample": False,
			"return_full_text": False,
		}
		self.inference_args = {**default_args, **(inference_args or {})}

	def _format_examples(self) -> str:
		if not self.conv_examples:
			return "Seller: Hi there!\nBuyer: Hello, I'm interested in the item."
		return "\n\n".join(
			exp.to_string_rep(keep_sys_da=True, keep_user_da=True)
			for exp in self.conv_examples
		)

	def _build_prompt(self, state: DialogSession, dialog_act: str) -> str:
		history = state.to_string_rep(
			keep_sys_da=True,
			keep_user_da=True,
			max_turn_to_display=self.max_hist_num_turns,
		)
		da_instruction = self.da_prompts_mapping.get(dialog_act, "")
		parts = [
			self.task_prompt,
			self.context_snippet,
			"Conversation so far:",
			history or "Seller: Hi there!\nBuyer: Hello.",
			f"Next seller action should follow dialog act [{dialog_act}] {da_instruction}",
			"Seller:",
		]
		return "\n\n".join(part for part in parts if part).strip()

	def _strip_tags(self, text: str) -> str:
		# remove leading [tag] artifacts the model might echo
		while True:
			match = TAG_PREFIX.match(text)
			if not match:
				break
			text = text[match.end() :].lstrip()
		return text

	def _clean_response(self, data) -> str:
		for resp in data:
			text = (resp.get("generated_text") or "").strip()
			if not text:
				continue
			if text.lower().startswith(f"{NegotiationGame.SYS.lower()}:"):
				text = text.split(":", 1)[1].strip()
			text = self._strip_tags(text)
			if text:
				return text
		return "Let's keep the conversation moving."

	def get_utterance(self, state: DialogSession, action: int) -> str:
		dialog_act = self.dialog_acts[action]
		prompt = self._build_prompt(state, dialog_act)
		log_prompt(f"[SELLER_MODEL]\n{prompt}")
		data = self.backbone_model.generate(prompt, **self.inference_args)
		reply = self._clean_response(data)
		logger.debug("Seller act=%s reply=%s", dialog_act, reply)
		return reply

	def get_utterance_batched(self, state: DialogSession, action: int, batch: int) -> List[str]:
		dialog_act = self.dialog_acts[action]
		prompt = self._build_prompt(state, dialog_act)
		args = {**self.inference_args}
		args["num_return_sequences"] = batch
		args["do_sample"] = True
		args["temperature"] = max(args.get("temperature", 0.7), 0.9)
		data = self.backbone_model.generate(prompt, **args)
		replies = [self._clean_response([resp]) for resp in data]
		return replies

	def get_utterance_w_da(self, state: DialogSession, action):
		raise NotImplementedError("SellerModel should only be used as the system agent.")

	def get_utterance_w_da_from_batched_states(self, states, action=None):
		raise NotImplementedError("SellerModel should only be used as the system agent.")


__all__ = ["SellerModel"]
