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
			PersuasionGame.S_LogicalAppeal:	 			"The Persuader use of reasoning and evidence to convince the Persuadee.",
			PersuasionGame.S_TaskRelatedInquiry:	 	"The Persuader asks about the Persuadee's knowledge or opinion related to Save the Children.",
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
		self.task_prompt = """
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children.
		The following is an example conversation between a Persuader and a Persuadee about Save the Children.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between Persuader (you) and another Persuadee.\nThe Persuader greets the persuadee."
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
	
	def get_utterance(self, state:DialogSession, action:int) -> str:
		return self.get_utterance_batched(state, action, batch=1)[0]
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int=3) -> List[str]:
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{PersuasionGame.USR}: Hello.\n{da_prompt}'})
		else:
			assert(state[-1][0] == PersuasionGame.USR)
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
		gen_args = {
			**self.inference_args,
			"num_return_sequences": batch,  # this will be changed to n inside chat_generate
		}
		data = self.backbone_model.chat_generate(messages, **gen_args)
		sys_resps = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{PersuasionGame.SYS}:", user_role=f"{PersuasionGame.USR}:"
		)
		return sys_resps

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError

__all__ = [
	"PersuaderModel",
	"PersuaderChatModel",
]
