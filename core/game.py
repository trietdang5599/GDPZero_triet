import numpy as np
import logging

from core.gen_models import DialogModel
from core.helpers import DialogSession
from abc import ABC, abstractmethod
from typing import List


logger = logging.getLogger(__name__)


class DialogGame(ABC):
	def __init__(self, 
			system_name:str, system_agent:DialogModel, 
			user_name: str, user_agent:DialogModel):
		self.SYS = system_name
		self.system_agent = system_agent
		self.USR = user_name
		self.user_agent = user_agent
		return

	@staticmethod
	@abstractmethod
	def get_game_ontology() -> dict:
		"""returns game related information such as dialog acts, slots, etc.
		"""
		raise NotImplementedError

	def init_dialog(self) -> DialogSession:
		# [(sys_act, sys_utt, user_act, user_utt), ...]
		return DialogSession(self.SYS, self.USR)

	def get_next_state(self, state:DialogSession, action) -> DialogSession:
		next_state = state.copy()

		sys_utt = self.system_agent.get_utterance(next_state, action)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		next_state.add_single(state.SYS, sys_da, sys_utt)
		
		# state in user's perspective
		user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None)  # user just reply
		next_state.add_single(state.USR, user_da, user_resp)
		return next_state
	
	def get_next_state_batched(self, state:DialogSession, action, batch=3) -> List[DialogSession]:
		all_next_states = [state.copy() for _ in range(batch)]

		sys_utts = self.system_agent.get_utterance_batched(state.copy(), action, batch)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		for i in range(batch):
			all_next_states[i].add_single(state.SYS, sys_da, sys_utts[i])
		
		# state in user's perspective
		user_das, user_resps = self.user_agent.get_utterance_w_da_from_batched_states(all_next_states, None)  # user just reply
		for i in range(batch):
			all_next_states[i].add_single(state.USR, user_das[i], user_resps[i])
		return all_next_states

	def display(self, state:DialogSession):
		string_rep = state.to_string_rep(keep_sys_da=True, keep_user_da=True)
		print(string_rep)
		return

	@abstractmethod
	def get_dialog_ended(self, state) -> float:
		"""returns 0 if not ended, then (in general) 1 if system success, -1 if failure 
		"""
		raise NotImplementedError
	
class PersuasionGame(DialogGame):
	SYS = "Persuader"
	USR = "Persuadee"

	S_CredibilityAppeal = "credibility appeal"
	S_EmotionAppeal = "emotion appeal"
	S_PropositionOfDonation = "proposition of donation"
	S_LogicalAppeal = "logical appeal"
	S_TaskRelatedInquiry = "task related inquiry"
	S_Greeting = "greeting"
	S_Other = "other"

	U_NoDonation = "no donation"
	U_NegativeReaction = "negative reaction"
	U_Neutral = "neutral"
	U_PositiveReaction = "positive reaction"
	U_Donate = "donate"

	def __init__(self, system_agent:DialogModel, user_agent:DialogModel, 
			max_conv_turns=15):
		super().__init__(PersuasionGame.SYS, system_agent, PersuasionGame.USR, user_agent)
		self.max_conv_turns = max_conv_turns
		return

	@staticmethod
	def get_game_ontology() -> dict:
		return {
			"system": {
				"dialog_acts": [
					PersuasionGame.S_Greeting, PersuasionGame.S_CredibilityAppeal, PersuasionGame.S_EmotionAppeal,
					PersuasionGame.S_PropositionOfDonation, PersuasionGame.S_LogicalAppeal,
					PersuasionGame.S_TaskRelatedInquiry,
					PersuasionGame.S_Other
				],
			},
			"user": {
				"dialog_acts": [
					PersuasionGame.U_NoDonation, PersuasionGame.U_NegativeReaction, PersuasionGame.U_Neutral,
					PersuasionGame.U_PositiveReaction, PersuasionGame.U_Donate
				]
			}
		}

	def get_dialog_ended(self, state) -> float:
		# check persuadee actions first
		for turn_idx, (role, da, utt) in enumerate(state):
			if da == PersuasionGame.U_Donate:
				logger.debug(
					"Dialog ended with donate (turn=%s, role=%s, utt=%s)",
					turn_idx,
					role,
					utt,
				)
				return 1.0
			if da == PersuasionGame.U_NoDonation:
				logger.debug(
					"Dialog ended with no-donation (turn=%s, role=%s, utt=%s)",
					turn_idx,
					role,
					utt,
				)
				return -1.0

		# fallback to max turn termination if nothing triggered
		if len(state) >= self.max_conv_turns:
			logger.debug(
				"Dialog ended with persuasion failure (reason=max_turns, turns=%s, last_state=%s)",
				len(state),
				state[-1] if state else None,
			)
			return -1.0
		return 0.0

	@staticmethod
	def map_user_da(raw_da: str) -> str:
		if raw_da == "disagree-donation":
			return PersuasionGame.U_NoDonation
		if raw_da == "negative-reaction-to-donation":
			return PersuasionGame.U_NegativeReaction
		if raw_da == "positive-reaction-to-donation":
			return PersuasionGame.U_PositiveReaction
		if raw_da == "agree-donation":
			return PersuasionGame.U_Donate
		return PersuasionGame.U_Neutral
	


class EmotionalSupportGame(PersuasionGame):
	pass


class NegotiationGame(DialogGame):
	SYS = "Seller"
	USR = "Buyer"

	S_INTRO = "seller-intro"
	S_INIT_PRICE = "seller-init-price"
	S_INFORM = "seller-inform"
	S_OFFER = "seller-offer"
	S_COUNTER = "seller-counter-price"
	S_VAGUE = "seller-vague-price"
	S_INSIST = "seller-insist"
	S_ACCEPT = "seller-accept"
	S_REJECT = "seller-reject"
	S_QUIT = "seller-quit"
	S_OTHER = "seller-other"

	B_GREETING = "buyer-greeting"
	B_INQUIRY = "buyer-inquiry"
	B_COUNTER = "buyer-counter-price"
	B_OFFER = "buyer-offer"
	B_ACCEPT = "buyer-accept"
	B_REJECT = "buyer-reject"
	B_QUIT = "buyer-quit"
	B_DISAGREE = "buyer-disagree"
	B_AGREE = "buyer-agree"
	B_OTHER = "buyer-other"

	def __init__(self, system_agent: DialogModel, user_agent: DialogModel, max_conv_turns: int = 20):
		super().__init__(NegotiationGame.SYS, system_agent, NegotiationGame.USR, user_agent)
		self.max_conv_turns = max_conv_turns

	@staticmethod
	def get_game_ontology() -> dict:
		return {
			"system": {"dialog_acts": [
				NegotiationGame.S_INTRO,
				NegotiationGame.S_INIT_PRICE,
				NegotiationGame.S_INFORM,
				NegotiationGame.S_OFFER,
				NegotiationGame.S_COUNTER,
				NegotiationGame.S_VAGUE,
				NegotiationGame.S_INSIST,
				NegotiationGame.S_ACCEPT,
				NegotiationGame.S_REJECT,
				NegotiationGame.S_QUIT,
				NegotiationGame.S_OTHER,
			]},
			"user": {"dialog_acts": [
				NegotiationGame.B_GREETING,
				NegotiationGame.B_INQUIRY,
				NegotiationGame.B_COUNTER,
				NegotiationGame.B_OFFER,
				NegotiationGame.B_ACCEPT,
				NegotiationGame.B_REJECT,
				NegotiationGame.B_QUIT,
				NegotiationGame.B_DISAGREE,
				NegotiationGame.B_AGREE,
				NegotiationGame.B_OTHER,
			]},
		}

	def get_dialog_ended(self, state: DialogSession) -> float:
		for role, da, _utt in state:
			if role == NegotiationGame.USR and da == NegotiationGame.B_ACCEPT:
				return 1.0
			if da in {
				NegotiationGame.B_REJECT,
				NegotiationGame.B_QUIT,
				NegotiationGame.S_REJECT,
				NegotiationGame.S_QUIT,
			}:
				return -1.0
		if len(state) >= self.max_conv_turns:
			return -1.0
		return 0.0

	@staticmethod
	def map_craigslist_intent(intent: str, role: str) -> str:
		intent_norm = (intent or "").strip().lower()
		if role == NegotiationGame.SYS:
			mapper = {
				"intro": NegotiationGame.S_INTRO,
				"init-price": NegotiationGame.S_INIT_PRICE,
				"inform": NegotiationGame.S_INFORM,
				"offer": NegotiationGame.S_OFFER,
				"counter-price": NegotiationGame.S_COUNTER,
				"vague-price": NegotiationGame.S_VAGUE,
				"insist": NegotiationGame.S_INSIST,
				"accept": NegotiationGame.S_ACCEPT,
				"agree": NegotiationGame.S_ACCEPT,
				"reject": NegotiationGame.S_REJECT,
				"quit": NegotiationGame.S_QUIT,
				"disagree": NegotiationGame.S_REJECT,
			}
		else:
			mapper = {
				"intro": NegotiationGame.B_GREETING,
				"inquiry": NegotiationGame.B_INQUIRY,
				"inform": NegotiationGame.B_AGREE,
				"offer": NegotiationGame.B_OFFER,
				"counter-price": NegotiationGame.B_COUNTER,
				"accept": NegotiationGame.B_ACCEPT,
				"agree": NegotiationGame.B_ACCEPT,
				"reject": NegotiationGame.B_REJECT,
				"quit": NegotiationGame.B_QUIT,
				"disagree": NegotiationGame.B_DISAGREE,
			}
		return mapper.get(intent_norm, NegotiationGame.S_OTHER if role == NegotiationGame.SYS else NegotiationGame.B_OTHER)
