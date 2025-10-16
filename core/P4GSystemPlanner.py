import logging
import numpy as np

from typing import List, Tuple
from core.helpers import DialogSession
from core.gen_models import GenerationModel
from core.game import PersuasionGame
from abc import ABC, abstractmethod
# from collections import Counter


logger = logging.getLogger(__name__)


class DialogPlanner(ABC):
	@abstractmethod
	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		pass

	@abstractmethod
	def predict(self, state) -> "Tuple[np.ndarray, float]":
		# returns a prob and value
		pass


class P4GSystemPlanner(DialogPlanner):
	def __init__(self, 
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns, 
			generation_model:GenerationModel, 
			conv_examples: List[DialogSession] = []) -> None:
		super().__init__()
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns  # used in prompting next da
		self.user_dialog_acts = user_dialog_acts
		self.user_max_hist_num_turns = user_max_hist_num_turns  # used in heuristic function
		self.conv_examples = conv_examples
		self.generation_model = generation_model
		self.smoothing = 1.0
		self.task_prompt = f"""
		The following is background information about Save the Children. 
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		The Persuader can choose amongst the following actions during a conversation:
		{" ".join([f"[{da}]" for da in dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()

		self.inf_args = {
			"max_new_tokens": 8,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}
		return

	def process_exp(self, keep_sys_da=True, keep_user_da=False):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_sys_da=keep_sys_da, keep_user_da=keep_user_da) + "\n"
		return prompt_exps.strip()

	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		turn = len(state)
		if turn < 1:
			return np.array([1 if da == PersuasionGame.S_Greeting else 0 for da in self.dialog_acts])
		return np.array([1 for _ in self.dialog_acts])

	def get_utterance(self, state, action) -> str:
		return ""  # should not be called

	def _get_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def predict(self, state:DialogSession) -> "Tuple[np.ndarray, float]":
		# test k times and compute prob. See num_return_sequences in the API
		# the value would be our objective function
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			Persuader:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{state.to_string_rep(keep_sys_da=True)}
			Persuader:
			"""
		prompt = prompt.replace("\t", "").strip()
		logger.debug(prompt)
		data = self.generation_model.generate(prompt, **self.inf_args)
		sampled_das = self._get_generated_da(data)
		logger.debug(f"sampled das: {sampled_das}")
		# convert to prob distribution
		prob = np.zeros(len(self.dialog_acts))
		prob += self.smoothing
		for da in sampled_das:
			prob[self.dialog_acts.index(da)] += 1
		prob /= prob.sum()
		v = self.heuristic(state)
		return prob, v

	def _get_user_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.user_dialog_acts:
				pred_da.append(found_da)
		return pred_da

	# compute heuristic value for a given state (Persuadee state)
	# e.g: sample_das = [U_Neutral, U_PositiveReaction, U_NegativeReaction, U_PositiveReaction] -> v = (0.0 + 0.5 - 0.5 + 0.5) / 4 = 0.125
	def heuristic(self, state:DialogSession) -> float:
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == PersuasionGame.USR)
		prompt = f"""
		The following is background information about task. 
		The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		The Persuadee can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.user_dialog_acts])}
		The following is a conversation between a Persuader and	a Persuadee about a charity called Save the Children. 
		{self.process_exp(keep_sys_da=False, keep_user_da=True)}
		The following is a new conversation between another Persuader and Persuadee.
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.user_max_hist_num_turns)}
		Persuader: Would you be interested in donating to Save the Children?
		Persuadee:
		"""
		prompt = prompt.replace("\t", "").strip()

		inf_args = {
			"max_new_tokens": 8,
			"temperature": 1.1,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 10,
		}
		data = self.generation_model.generate(prompt, **inf_args)
		sampled_das = self._get_user_generated_da(data)

		logger.debug(f"persuadee prompt: {prompt}")
		logger.debug(f"sampled das: {sampled_das}")

		# heuristic score
		score = []
		for da in sampled_das:
			if da == PersuasionGame.U_NoDonation:
				score.append(-1.0)
			elif da == PersuasionGame.U_NegativeReaction:
				score.append(-0.5)
			elif da == PersuasionGame.U_Neutral:
				score.append(0.0)
			elif da == PersuasionGame.U_PositiveReaction:
				score.append(0.5)
			elif da == PersuasionGame.U_Donate:
				score.append(1.0)
		v = 0.0 if len(score) == 0 else np.mean(score)
		logger.debug(f"sampled das to v: {v}")
		return float(v)
	

class P4GChatSystemPlanner(P4GSystemPlanner):
	def __init__(self, 
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns, 
			generation_model:GenerationModel, 
			conv_examples: List[DialogSession] = []) -> None:
		super().__init__(
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns,
			generation_model, conv_examples
		)
		self.task_prompt = f"""
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children. You can choose amongst the following actions during a conversation:
		{" ".join([f"[{da}]" for da in dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about Save the Children.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between Persuader (you) and a Persuadee."
		self.prompt_examples = self.process_chat_exp(new_task_prompt=self.new_task_prompt)

		self.inf_args = {
			"max_new_tokens": 12,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}
		return
	
	def process_chat_exp(self, 
			new_task_prompt,
			assistant_role=PersuasionGame.SYS,
			keep_sys_da=True, keep_user_da=False):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_chat_exp(exp, keep_sys_da, keep_user_da, assistant_role)
			prompt_exps.append({
				"role":"system", "content": new_task_prompt
			})
		return prompt_exps[:-1]

	def __proccess_chat_exp(self,
			exp:DialogSession, 
			keep_sys_da, keep_user_da,
			assistant_role=PersuasionGame.SYS,
			max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# P4G dataset starts with the system/Persuader
		assert(exp[0][0] == PersuasionGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		# init with user
		# if assistant_role == PersuasionGame.SYS:
		# 	if keep_user_da:
		# 		prompt_messages.append({
		# 			"role": "user",
		# 			"content": f"{PersuasionGame.USR}: [{PersuasionGame.U_Neutral}] Hello.".strip()
		# 		})
		# 	else:
		# 		prompt_messages.append({
		# 			"role": "user",
		# 			"content": f"{PersuasionGame.USR}: Hello.".strip()
		# 		})
		# all the rest
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			# if assistant is the Persuader, then current data is also Persuader -> then it is of role "system"
			if role == PersuasionGame.SYS:
				if keep_sys_da:
					content = f"{role}: [{da}] {utt}".strip()
				else:
					content = f"{role}: {utt}".strip()
				if assistant_role == PersuasionGame.SYS:
					prompt_role = "assistant"
				else:
					prompt_role = "user"
			else:
				if keep_user_da:
					content = f"{role}: [{da}] {utt}".strip()
				else:
					content = f"{role}: {utt}".strip()
				if assistant_role == PersuasionGame.USR:
					prompt_role = "assistant"
				else:
					prompt_role = "user"
			
			prompt_messages.append({
				"role": prompt_role,
				"content": content
			})
		return prompt_messages

	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		turn = len(state)
		if turn < 1:
			return np.array([1 if da == PersuasionGame.S_Greeting else 0 for da in self.dialog_acts])
		return np.array([1 for _ in self.dialog_acts])

	def get_utterance(self, state, action) -> str:
		return ""  # should not be called

	def _get_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def predict(self, state:DialogSession) -> "Tuple[np.ndarray, float]":
		# test k times and compute prob. See num_return_sequences in the API
		# the value would be our objective function
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{PersuasionGame.USR}: Hello.'})
		else:
			assert(state[-1][0] == PersuasionGame.USR)
			messages += self.__proccess_chat_exp(state, keep_sys_da=True, keep_user_da=False)
		# produce a response
		data = self.generation_model.chat_generate(messages, **self.inf_args)

		sampled_das = self._get_generated_da(data)
		logger.debug(f"sampled das: {sampled_das}")
		# convert to prob distribution
		prob = np.zeros(len(self.dialog_acts))
		prob += self.smoothing
		for da in sampled_das:
			prob[self.dialog_acts.index(da)] += 1
		prob /= prob.sum()
		v = self.heuristic(state)
		return prob, v

	def _get_user_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.user_dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def heuristic(self, state:DialogSession) -> float:
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == PersuasionGame.USR)
		user_task_prompt = f"""
		You are a persuadee. A Persuader is trying to persuade you to donate to a charity called Save the Children.
		You can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.user_dialog_acts])}
		The following is a new conversation between a Persuader and a Persuadee (you).
		""".replace("\t", "").strip()
		user_new_task_prompt = "The following is a new conversation between a Persuader and a Persuadee (you)."

		messages = [
			{'role': 'system', 'content': user_task_prompt},
			*self.process_chat_exp(new_task_prompt=user_new_task_prompt, assistant_role=PersuasionGame.USR, keep_sys_da=False, keep_user_da=True),
			{'role': 'system', 'content': user_new_task_prompt}
		]
		messages += self.__proccess_chat_exp(state, assistant_role=PersuasionGame.USR, keep_sys_da=False, keep_user_da=True)
		messages.append({
			'role': 'user', 'content': f'{PersuasionGame.SYS}: Would you be interested in donating to Save the Children?'
		})

		inf_args = {
			"max_new_tokens": 12,
			"temperature": 1.1,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 10,
		}
		data = self.generation_model.chat_generate(messages, **inf_args)
		sampled_das = self._get_user_generated_da(data)

		logger.debug(f"persuadee prompt: {messages}")
		logger.debug(f"sampled das: {sampled_das}")

		# heuristic score
		score = []
		for da in sampled_das:
			if da == PersuasionGame.U_NoDonation:
				score.append(-1.0)
			elif da == PersuasionGame.U_NegativeReaction:
				score.append(-0.5)
			elif da == PersuasionGame.U_Neutral:
				score.append(0.0)
			elif da == PersuasionGame.U_PositiveReaction:
				score.append(0.5)
			elif da == PersuasionGame.U_Donate:
				score.append(1.0)
		v = 0.0 if len(score) == 0 else np.mean(score)
		logger.debug(f"sampled das to v: {v}")
		return float(v)

__all__ = [
	"DialogPlanner",
	"P4GSystemPlanner",
	"P4GChatSystemPlanner",
]

