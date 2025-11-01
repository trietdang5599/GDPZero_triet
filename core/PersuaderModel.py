import logging
import re

from typing import List, Tuple

from core.helpers import DialogSession
from core.gen_models import GenerationModel, DialogModel, LocalModel
from core.game import PersuasionGame
from utils.utils import log_prompt, format_messages_for_log
from utils.dialog_acts import SYSTEM_DIALOG_ACT_DEFINITIONS


logger = logging.getLogger(__name__)

class PersuaderModel(DialogModel):
	def __init__(self,
			dialog_acts:List[str],
			backbone_model:GenerationModel,
			max_hist_num_turns: int = 5,
			conv_examples: List[DialogSession] = [],
			inference_args: dict = {}):
		super().__init__()
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.max_hist_num_turns = max_hist_num_turns
		# prompts and DAs
		self.da_prompts_mapping = {
			da: desc for da, desc in SYSTEM_DIALOG_ACT_DEFINITIONS.items() if da in dialog_acts
		}
		# only allow da that has the mapping
		self.dialog_acts = [da for da in dialog_acts if da in self.da_prompts_mapping]
		
		logger.debug(self.dialog_acts)
		self.task_prompt = f"""
		The following is background information about Save the Children. 
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()
		self.response_instruction = (
			"Respond politely with at least one complete sentence that advances the persuasion objective."
		)
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise tree will never go to the next level
			"return_full_text": False,
			**inference_args
		}
		return

	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_exp(exp) + "\n"
		return prompt_exps.strip()

	def _normalize_response(self, text: str) -> str:
		return " ".join((text or "").strip().split()).lower()

	def _response_was_used(self, state: DialogSession, response: str) -> bool:
		if not response:
			return False
		target = self._normalize_response(response)
		if not target:
			return False
		for role, _da, utt in reversed(state):
			if role != PersuasionGame.SYS:
				continue
			if self._normalize_response(utt) == target:
				return True
		return False

	def _generate_responses(self, prompt: str, sampling: bool, num_return_sequences: int) -> List[str]:
		gen_args = {**self.inference_args}
		gen_args["num_return_sequences"] = max(1, int(num_return_sequences))
		if sampling:
			gen_args["do_sample"] = True
			gen_args.setdefault("temperature", 0.9)
			gen_args.setdefault("top_p", 0.9)
		else:
			gen_args["do_sample"] = False
		try:
			data = self.backbone_model.generate(prompt, **gen_args)
			return self.backbone_model._cleaned_resp(data, prompt)
		except Exception as exc:  # pragma: no cover - robustness
			logger.warning("Persuader generation failed (sampling=%s): %s", sampling, exc)
			return []

	def __proccess_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		prompt_exp = ""
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			
			if role == PersuasionGame.SYS:
				prompt_exp += f"{self.da_prompts_mapping[da]}\n{role}: {utt}\n"
			else:
				prompt_exp += f"{role}: {utt}\n"
		return prompt_exp.strip()
	
	def get_utterance(self, state:DialogSession, action:int) -> str:
		# planner gives an action, state is history, you need to produce a response accrd to the action
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		prompt = self._build_prompt(state, da_prompt)
		log_prompt(f"[PERSUADER]\n{prompt}")
		# First attempt: deterministic decoding
		candidates = self._generate_responses(prompt, sampling=False, num_return_sequences=1)
		primary_resp = candidates[0] if candidates else ""
		if primary_resp and not self._response_was_used(state, primary_resp):
			return primary_resp

		# Fallback: stochastic decoding to avoid repetition
		logger.debug("Detected repeated or empty response; resampling Persuader utterance.")
		for candidate in self._generate_responses(prompt, sampling=True, num_return_sequences=3):
			if candidate and not self._response_was_used(state, candidate):
				return candidate
		# If all else fails, return original (even if repeated) to avoid empty turn.
		return primary_resp

	def _build_prompt(self, state: DialogSession, da_prompt: str) -> str:
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			{self.response_instruction}
			{da_prompt}
			Persuader:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{self.__proccess_exp(state, max_hist_num_turns=self.max_hist_num_turns)}
			{self.response_instruction}
			{da_prompt}
			Persuader:
			"""
		return prompt.replace("\t", "").strip()

	def get_utterance_batched(
		self,
		state: DialogSession,
		action: int,
		batch: int = 3,
		sampling: bool | None = None,
	) -> List[str]:
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		prompt = self._build_prompt(state, da_prompt)
		if sampling is None:
			sampling = batch > 1
		return self._generate_responses(prompt, sampling=sampling, num_return_sequences=batch)

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError
	

class PersuaderChatModel(PersuaderModel):
	def __init__(self,
			dialog_acts:List[str],
			backbone_model:GenerationModel,
			max_hist_num_turns: int = 5,
			conv_examples: List[DialogSession] = [],
			inference_args: dict = {}):
		super().__init__(
			dialog_acts=dialog_acts,
			backbone_model=backbone_model,
			max_hist_num_turns=max_hist_num_turns,
			conv_examples=conv_examples,
			inference_args=inference_args
		)
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise tree will never go to the next level, unless you do OpenLoop search
			"return_full_text": False,
			**inference_args
		}
		self.task_prompt = """
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children.
		The following is an example conversation between a Persuader and a Persuadee about Save the Children.
		""".replace("\t", "").strip()
		self.new_task_prompt = (
			"The following is a new conversation between Persuader (you) and another Persuadee.\n"
			"Always answer politely with one or more complete sentences that advance the persuasion goal. (NEVER answer empty or meaningless)"
		)
		self.prompt_examples = self.process_chat_exp()
		return

	def process_chat_exp(self):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_chat_exp(exp)
			prompt_exps.append({
				"role":"system", "content": self.new_task_prompt
			})
		return prompt_exps[:-1]

	def __proccess_chat_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# P4G dataset starts with the system
		assert(exp[0][0] == PersuasionGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		
		next_sys_da = PersuasionGame.S_Greeting
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			if role == PersuasionGame.SYS:
				prompt_messages.append({
					"role": "assistant",
					"content": f"{role}: {utt}".strip()
				})
			else:
				if i+1 < len(exp.history):
					next_sys_da = exp[i+1][1]
					prompt_messages.append({
						"role": "user",
						"content": f"{role}: {utt}\n{self.da_prompts_mapping[next_sys_da]}".strip()
					})
				else:
					prompt_messages.append({
						"role": "user",
						"content": f"{role}: {utt}".strip()
				})
		return prompt_messages
	
	def _build_chat_messages(self, state: DialogSession, action: int) -> List[dict]:
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		messages: List[dict] = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{PersuasionGame.USR}: Hello.\n{da_prompt}'})
		else:
			assert state[-1][0] == PersuasionGame.USR
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
			messages.append({'role': 'system', 'content': da_prompt})
		return messages

	def _chat_generate(self, messages: List[dict], sampling: bool, num_sequences: int) -> List[str]:
		gen_args = {**self.inference_args}
		gen_args["num_return_sequences"] = max(1, int(num_sequences))
		if sampling:
			gen_args["do_sample"] = True
			gen_args.setdefault("temperature", 0.9)
			gen_args.setdefault("top_p", 0.9)
		else:
			gen_args["do_sample"] = False
		try:
			data = self.backbone_model.chat_generate(messages, **gen_args)
			return self.backbone_model._cleaned_chat_resp(
				data,
				assistant_role=f"{PersuasionGame.SYS}:",
				user_role=f"{PersuasionGame.USR}:",
			)
		except Exception as exc:  # pragma: no cover
			logger.warning("Persuader chat generation failed (sampling=%s): %s", sampling, exc)
			return []

	def get_utterance(self, state:DialogSession, action:int) -> str:
		messages = self._build_chat_messages(state, action)
		log_prompt(f"[PERSUADER_CHAT]\n{format_messages_for_log(messages)}")

		candidates = self._chat_generate(messages, sampling=False, num_sequences=1)
		primary_resp = candidates[0] if candidates else ""
		if primary_resp and not self._response_was_used(state, primary_resp):
			return primary_resp

		logger.debug("Resampling Persuader chat response to avoid repetition.")
		for candidate in self._chat_generate(messages, sampling=True, num_sequences=3):
			if candidate and not self._response_was_used(state, candidate):
				return candidate
		return primary_resp or ""
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int=3, sampling: bool | None = None) -> List[str]:
		messages = self._build_chat_messages(state, action)
		log_prompt(f"[PERSUADER_CHAT]\n{format_messages_for_log(messages)}")
		if sampling is None:
			sampling = batch > 1
		return self._chat_generate(messages, sampling=sampling, num_sequences=batch)

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError

__all__ = [
	"PersuaderModel",
	"PersuaderChatModel",
]
