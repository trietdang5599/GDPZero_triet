import logging
import re

from typing import List, Tuple

from core.helpers import DialogSession
from core.gen_models import GenerationModel, DialogModel, LocalModel
from core.game import PersuasionGame


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
			PersuasionGame.S_Greeting:	 				"The Persuader greets the Persuadee.",
			# start of persuasion strategies
			PersuasionGame.S_CredibilityAppeal:	 		"The Persuader establishes credibility of Save the Children by citing its impact.",
			PersuasionGame.S_EmotionAppeal:	 			"The Persuader uses an emotion appeal to convince the Persuadee.",
   			PersuasionGame.S_PersonalStory:				"The Persuader shares a personal story related to Save the Children to build rapport.",
			PersuasionGame.S_LogicalAppeal:	 			"The Persuader use of reasoning and evidence to convince the Persuadee.",
			PersuasionGame.S_FootInTheDoor:			"The Persuader makes a small request first to increase the chance of a later donation.",
			PersuasionGame.S_SelfModeling:			"The Persuader describes their own generous behaviour to encourage imitation.",
			PersuasionGame.S_TaskRelatedInquiry:	 	"The Persuader asks about the Persuadee's knowledge or opinion related to Save the Children.",
			PersuasionGame.S_SourceRelatedInquiry:		"The Persuader asks about where the Persuadee gets information or whom they trust.",
			PersuasionGame.S_PersonalRelatedInquiry:	"The Persuader asks personal questions to better understand the Persuadee's motivations.",
			PersuasionGame.S_NeutralToInquiry:		"The Persuader asks neutral follow-up questions without pushing for donation yet.",
			PersuasionGame.S_PropositionOfDonation:	 	"The Persuader asks if the Persuadee would like to make a small donation.",
			# end of persuasion strategies
   
			PersuasionGame.S_Other:	 					"The Persuader responds to the Persuadee without using any persuaive strategy.",
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
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise tree will never go to the next level
			"return_full_text": False,
			**inference_args
		}
		if isinstance(backbone_model, LocalModel):
			self.inference_args.setdefault("no_repeat_ngram_size", 3)
			self.inference_args["repetition_penalty"] = max(1.1, self.inference_args.get("repetition_penalty", 1.0))
			self.inference_args.setdefault("temperature", 0.7)
		return

	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_exp(exp) + "\n"
		return prompt_exps.strip()

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
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			{da_prompt}
			Persuader:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{self.__proccess_exp(state, max_hist_num_turns=self.max_hist_num_turns)}
			{da_prompt}
			Persuader:
			"""
		prompt = prompt.replace("\t", "").strip()
		# produce a response
		data = self.backbone_model.generate(prompt, **self.inference_args)
		sys_resp = self.backbone_model._cleaned_resp(data, prompt)[0]  # TODO
		return sys_resp

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
		if isinstance(backbone_model, LocalModel):
			self.inference_args.setdefault("no_repeat_ngram_size", 3)
			self.inference_args["repetition_penalty"] = max(1.1, self.inference_args.get("repetition_penalty", 1.0))
			self.inference_args.setdefault("temperature", 0.7)
		self.task_prompt = """
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children.
		The following is an example conversation between a Persuader and a Persuadee about Save the Children.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between Persuader (you) and another Persuadee.\nThe Persuader greets the persuadee."
		self.prompt_examples = self.process_chat_exp()
		return

	def _format_da_instruction(self, da: str) -> str:
		return (
			"Instruction for the Persuader response. "
			f"Goal: {self.da_prompts_mapping[da]} "
			"Return exactly one line wrapped in <answer></answer>, where the content is `Persuader: <utterance>`. "
			"Inside <answer> do not include any additional XML tags, brackets, or role labels besides `Persuader:`. "
			"The utterance must sound natural, stay on topic, use at most three sentences, and avoid greetings like `Human.` "
			"Do not add narration, analysis, policy reminders, or any text outside the <answer> tag."
		)

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
				prompt_messages.append({
					"role": "user",
					"content": f"{role}: {utt}".strip()
				})
				if i + 1 < len(exp.history):
					next_sys_da = exp[i + 1][1]
					prompt_messages.append({
						"role": "system",
						"content": self._format_da_instruction(next_sys_da)
					})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action:int) -> str:
		return self.get_utterance_batched(state, action, batch=1)[0]

	@staticmethod
	def _sanitize_response(text: str) -> str:
		match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
		sanitized = match.group(1) if match else text
		sanitized = sanitized.strip()
		# strip any residual XML-like tags the model might emit (e.g. <Persuader>)
		sanitized = re.sub(r"</?[^>\s]+>", "", sanitized)
		# remove disclaimer-style suffixes the model may attach after the main sentence
		disclaimer_patterns = [
			r"[.?!]\s*disclaimer:.*",
			r"[.?!]\s*please note.*",
			r"[.?!]\s*remember.*",
			r"[.?!]\s*as an ai.*",
		]
		for pattern in disclaimer_patterns:
			sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE).rstrip()
		if "\n" in sanitized:
			sanitized = sanitized.splitlines()[0].strip()
		disclaimer_markers = [
			"you are an ai assistant",
			"user will you give you a task",
			"adhere to ethical guidelines",
			"complete the task as faithfully",
		]
		lower_sanitized = sanitized.lower()
		for marker in disclaimer_markers:
			idx = lower_sanitized.find(marker)
			if idx != -1:
				sanitized = sanitized[:idx].strip()
				break
		if not sanitized:
			return "Persuader: "
		if not sanitized.lower().startswith("persuader:"):
			sanitized = f"{sanitized}"
		return sanitized
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int=3) -> List[str]:
		da = self.dialog_acts[action]
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{PersuasionGame.USR}: Hello.'})
		else:
			assert(state[-1][0] == PersuasionGame.USR)
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
		messages.append({'role': 'system', 'content': self._format_da_instruction(da)})
		gen_args = {
			**self.inference_args,
			"num_return_sequences": batch,  # this will be changed to n inside chat_generate
		}
		data = self.backbone_model.chat_generate(messages, **gen_args)
		sys_resps = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{PersuasionGame.SYS}:", user_role=f"{PersuasionGame.USR}:"
		)
		return [self._sanitize_response(resp) for resp in sys_resps]

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError


__all__ = [
	"PersuaderModel",
	"PersuaderChatModel",
]
