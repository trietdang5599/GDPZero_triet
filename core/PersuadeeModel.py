import logging

from collections import Counter
from typing import List, Tuple

from core.helpers import DialogSession
from core.gen_models import GenerationModel, DialogModel
from core.game import PersuasionGame


logger = logging.getLogger(__name__)

class PersuadeeModel(DialogModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel, 
			conv_examples: List[DialogSession] = [], 
			max_hist_num_turns=5):
		super().__init__()
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns
		# prompts
		self.task_prompt = f"""
		The following is background information about task. 
		The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		The Persuadee can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()
		self.inference_args = inference_args
		return
	
	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_user_da=True) + "\n"
		return prompt_exps.strip()
	
	def get_utterance(self, state:DialogSession, action=None) -> str:
		assert(state[-1][0] == PersuasionGame.SYS)
		prompt = f"""
		{self.task_prompt}
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.max_hist_num_turns)}
		Persuadee:
		"""
		prompt = prompt.replace("\t", "").strip()
		# produce a response
		data = self.backbone_model.generate(prompt, **self.inference_args)
		user_resp = self.backbone_model._cleaned_resp(data, prompt)[0]
		return user_resp

	def get_utterance_w_da(self, state:DialogSession, action=None) -> "Tuple[str, str]":
		user_resp = self.get_utterance(state, action)
		# extract da
		start_idx = user_resp.find("[")
		end_idx = user_resp.find("]")
		if start_idx == -1 or end_idx == -1:
			da = PersuasionGame.U_Neutral
		else:
			da = user_resp[start_idx+1:end_idx]
			user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
			if da not in self.dialog_acts:
				da = PersuasionGame.U_Neutral
		return da, user_resp


class PersuadeeChatModel(PersuadeeModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel, 
			conv_examples: List[DialogSession] = [], 
			max_hist_num_turns=5):
		super().__init__(
			dialog_acts=dialog_acts,
			inference_args=inference_args,
			backbone_model=backbone_model,
			conv_examples=conv_examples,
			max_hist_num_turns=max_hist_num_turns
		)
		self.inference_args = inference_args
		self.task_prompt = f"""
		You are a persuadee. A Persuader is trying to persuade you to donate to a charity called Save the Children.
		You can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation between a Persuader and some Persuadee.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between a Persuader and a Persuadee (you). You may or may not want to donate to Save the Children."
		self.heuristic_args: dict = {
			"max_hist_num_turns": 2,
			"example_pred_turn": [[0, 2, 3, 4]]
		}
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
		
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			if role == PersuasionGame.SYS:
				prompt_messages.append({
					"role": "user",
					"content": f"{role}: {utt}".strip()
				})
			else:
				prompt_messages.append({
					"role": "assistant",  # assistant is the user simulator
					"content": f"{role}: [{da}] {utt}".strip()
				})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action=None) -> str:
		assert(state[-1][0] == PersuasionGame.SYS)  # next turn is user's turn
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)

		# produce a response
		data = self.backbone_model.chat_generate(messages, **self.inference_args)
		user_resp = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{PersuasionGame.USR}:", user_role=f"{PersuasionGame.SYS}:"
		)[0]
		return user_resp
	
	def get_utterance_from_batched_states(self, states:List[DialogSession], action=None) -> List[str]:
		assert(all([state[-1][0] == PersuasionGame.SYS for state in states]))
		all_prompts = []
		for state in states:
			messages = [
				{'role': 'system', 'content': self.task_prompt},
				*self.prompt_examples,
				{'role': 'system', 'content': self.new_task_prompt}
			]
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
			all_prompts.append(messages)
		# produce a response
		datas = self.backbone_model.chat_generate_batched(all_prompts, **self.inference_args)
		user_resps = []
		for data in datas:
			user_resp = self.backbone_model._cleaned_chat_resp(
				data, assistant_role=f"{PersuasionGame.USR}:", user_role=f"{PersuasionGame.SYS}:"
			)
			user_resps.append(user_resp[0])
		return user_resps
	
	def get_utterance_w_da_from_batched_states(self, states:List[DialogSession], action=None):
		gen_user_resps = self.get_utterance_from_batched_states(states, action)
		das = []
		user_resps = []
		# extract da
		for user_resp in gen_user_resps:
			start_idx = user_resp.find("[")
			end_idx = user_resp.find("]")
			if start_idx == -1 or end_idx == -1:
				da = PersuasionGame.U_Neutral
			else:
				da = user_resp[start_idx+1:end_idx]
				user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
				if da not in self.dialog_acts:
					da = PersuasionGame.U_Neutral
			das.append(da)
			user_resps.append(user_resp)
		return das, user_resps

	def __process_heuristics_chat_exp(self, dialog:DialogSession):
		if len(dialog) == 0:
			return []
		# assumes you start with the system
		# and ends with a user utterance to predict
		assert(dialog[0][0] == PersuasionGame.SYS)
		assert(dialog[-1][0] == PersuasionGame.USR)

		prompt_messages = []
		input_context = []
		answer_da = dialog[-1][1]
		for i, (role, da, utt) in enumerate(dialog):
			# if assistant is the Persuader, then current data is also Persuader -> then it is of role "system"
			# treat this as a task
			content = f"{role}: {utt}".strip()
			input_context.append(content)
		input_context.append(f"{dialog.USR} feeling:")

		prompt_q = "\n".join(input_context)
		prompt_messages.append({
			"role": 'user',
			"content": prompt_q
		})
		prompt_messages.append({
			"role": 'assistant',
			"content": f"{answer_da}"
		})
		return prompt_messages
	
	def __truncate_heuristics_dialog(self, dialog:DialogSession, pred_end_idx=-1):
		max_history_length = self.heuristic_args['max_hist_num_turns']
		if pred_end_idx == -1:
			pred_end_idx = len(dialog.history) - 1
		new_sys_start_idx = max(0, pred_end_idx - (max_history_length * 2 - 1))
		new_history = []
		for j, (role, da, utt) in enumerate(dialog):
			if j >= new_sys_start_idx:
				new_history.append((role, da, utt))
			if j == pred_end_idx:
				# user's utternace to predict
				break
		new_dialog_session = DialogSession(dialog.SYS, dialog.USR).from_history(new_history)
		return new_dialog_session
	
	def process_heurstics_chat_exp(self, new_task_prompt: str):
		prompt_exps = []
		for i, exp in enumerate(self.conv_examples):
			pred_end_turns: List[int] = self.heuristic_args['example_pred_turn'][i]
			# make a new dialogue session until that pred_idx with max max_history_length turns
			for pred_end_turn in pred_end_turns:
				pred_end_idx = pred_end_turn * 2 + 1
				new_dialog_session = self.__truncate_heuristics_dialog(exp, pred_end_idx)
				prompt_exps += self.__process_heuristics_chat_exp(new_dialog_session)
				prompt_exps.append({
					"role":"system", "content": new_task_prompt
				})
		return prompt_exps[:-1]

	def predict_da(self, state:DialogSession, never_end=True) -> str:
		# never_end=True  during real chat, let user choose to terminate, not this function
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == PersuasionGame.USR)

		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.process_heurstics_chat_exp(new_task_prompt=self.new_task_prompt),
			{'role': 'system', 'content': self.new_task_prompt}
		]
		new_dialog_session = self.__truncate_heuristics_dialog(state, -1)
		messages += self.__process_heuristics_chat_exp(new_dialog_session)[:-1]

		# majority vote, same as value function
		inf_args = {
			"max_new_tokens": 5,
			"temperature": 0.7,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 5,
		}
		datas = self.backbone_model.chat_generate(messages, **inf_args)
		# process into das
		sampled_das: list = []
		for resp in datas:
			user_da = resp['generated_text'].strip()
			if user_da not in self.dialog_acts:
				sampled_das.append(PersuasionGame.U_Neutral)
			if never_end:
				if user_da == PersuasionGame.U_Donate:
					sampled_das.append(PersuasionGame.U_PositiveReaction)
				elif user_da == PersuasionGame.U_NoDonation:
					sampled_das.append(PersuasionGame.U_NegativeReaction)
				else:
					sampled_das.append(user_da)
			else:
				sampled_das.append(user_da)
		logger.info(f"sampled das: {sampled_das}")
		# majority vote
		counted_das = Counter(sampled_das)
		user_da = counted_das.most_common(1)[0][0]
		return user_da

__all__ = [
	"PersuadeeModel",
	"PersuadeeChatModel",
]

