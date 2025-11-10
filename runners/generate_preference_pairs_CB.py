#!/usr/bin/env python3

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.CBSystemPlanner import CBSystemPlanner
from core.SellerModel import SellerModel
from core.BuyerModel import BuyerModel
from core.game import NegotiationGame
from core.helpers import DialogSession
from core.mcts import OpenLoopMCTSParallel
from core.model_factory import create_factor_llm
from utils.utils import (
	dotdict,
	export_preference_pair,
	get_preference_pair,
	set_determinitic_seed,
)


logger = logging.getLogger(__name__)

DEFAULT_CB_DATASET = PROJECT_ROOT / "data" / "CraigslistBargains" / "train.json"

_CB_DIALOG_CACHE: Dict[str, List[Tuple[str, dict]]] = {}
_CB_ANCHOR_POINTERS: Dict[str, int] = {}

class CBBuyerPlanner:
	def __init__(self, dialog_acts: Sequence[str], rng: Optional[random.Random] = None):
		self.dialog_acts = list(dialog_acts)
		self.rng = rng or random.Random()

	def select_action(self, state: DialogSession) -> str:
		if len(state) == 0:
			return NegotiationGame.B_GREETING if NegotiationGame.B_GREETING in self.dialog_acts else self.dialog_acts[0]
		last_turn = state[-1]
		last_role, last_da, _ = last_turn
		if last_role != NegotiationGame.SYS and len(state) >= 2:
			last_role, last_da, _ = state[-2]
		if last_da in {
			NegotiationGame.S_ACCEPT,
			NegotiationGame.S_QUIT,
			NegotiationGame.S_REJECT,
		}:
			return NegotiationGame.B_ACCEPT if last_da == NegotiationGame.S_ACCEPT else NegotiationGame.B_REJECT
		if last_da in {NegotiationGame.S_INIT_PRICE, NegotiationGame.S_OFFER, NegotiationGame.S_COUNTER}:
			return NegotiationGame.B_COUNTER if NegotiationGame.B_COUNTER in self.dialog_acts else NegotiationGame.B_INQUIRY
		if last_da in {NegotiationGame.S_INSIST, NegotiationGame.S_VAGUE}:
			return NegotiationGame.B_DISAGREE if NegotiationGame.B_DISAGREE in self.dialog_acts else NegotiationGame.B_COUNTER
		if last_da == NegotiationGame.S_INFORM:
			return NegotiationGame.B_INQUIRY
		return self.rng.choice(self.dialog_acts)


def classify_buyer_act(utterance: str, fallback: str) -> str:
	text = (utterance or "").lower()
	if any(keyword in text for keyword in ["deal", "i'll take", "sounds good", "accept"]):
		return NegotiationGame.B_ACCEPT
	if any(keyword in text for keyword in ["can't", "pass", "not interested", "reject"]):
		return NegotiationGame.B_REJECT
	if any(keyword in text for keyword in ["too high", "lower", "any chance", "budget", "counter"]):
		return NegotiationGame.B_COUNTER
	if any(keyword in text for keyword in ["question", "condition", "how old", "still available"]):
		return NegotiationGame.B_INQUIRY
	if any(keyword in text for keyword in ["thank", "hello", "hi"]):
		return NegotiationGame.B_GREETING
	if any(keyword in text for keyword in ["firm", "hard to justify", "hmm", "maybe"]):
		return NegotiationGame.B_DISAGREE
	return fallback


def _infer_cb_role(personal_role: Optional[str]) -> Optional[str]:
	if not personal_role:
		return None
	role = str(personal_role).strip().lower()
	if "seller" in role or "poster" in role:
		return NegotiationGame.SYS
	if "buyer" in role:
		return NegotiationGame.USR
	return None


def _resolve_agent_roles(record: dict) -> Dict[int, str]:
	roles: Dict[int, str] = {}
	scenario = record.get("scenario") or {}
	kbs = scenario.get("kbs") or []
	for idx, kb in enumerate(kbs):
		role = _infer_cb_role((kb or {}).get("personal", {}).get("Role"))
		if role:
			roles[idx] = role
	if not roles:
		# default heuristic: agent 1 -> seller, agent 0 -> buyer
		roles = {0: NegotiationGame.USR, 1: NegotiationGame.SYS}
	return roles


def _segment_cb_dialog(record: dict) -> List[Tuple[str, str, str]]:
	events = record.get("events")
	if not isinstance(events, list):
		return []
	roles = _resolve_agent_roles(record)

	segments: List[Tuple[str, str, str]] = []
	current_role: Optional[str] = None
	buffer: List[str] = []
	intent_buffer: Optional[str] = None

	for event in events:
		if not isinstance(event, dict):
			continue
		if event.get("action") != "message":
			continue
		role = roles.get(event.get("agent"))
		text = str(event.get("data") or "").strip()
		intent = (event.get("metadata") or {}).get("intent")
		if not role or not text:
			continue
		if role != current_role and buffer:
			segments.append((current_role, intent_buffer or "", " ".join(buffer).strip()))
			buffer = []
		buffer.append(text)
		intent_buffer = intent or intent_buffer
		current_role = role

	if buffer and current_role:
		segments.append((current_role, intent_buffer or "", " ".join(buffer).strip()))
	return segments


def _paired_segments(segments: Sequence[Tuple[str, str, str]]) -> List[Tuple[Tuple[str, str, str], Tuple[str, str, str]]]:
	pairs: List[Tuple[Tuple[str, str, str], Tuple[str, str, str]]] = []
	i = 0
	while i + 1 < len(segments):
		sys_seg = segments[i]
		usr_seg = segments[i + 1]
		if sys_seg[0] != NegotiationGame.SYS:
			i += 1
			continue
		if usr_seg[0] != NegotiationGame.USR:
			i += 1
			continue
		pairs.append((sys_seg, usr_seg))
		i += 2
	return pairs


def load_cb_dialogs(dataset_path: Optional[Path] = None) -> List[Tuple[str, dict]]:
	if dataset_path is None:
		resolved = DEFAULT_CB_DATASET.resolve()
	else:
		resolved = dataset_path.resolve()
	key = str(resolved)
	if key in _CB_DIALOG_CACHE:
		return _CB_DIALOG_CACHE[key]
	if not resolved.exists():
		logger.warning("Craigslist Bargains dataset not found at %s.", resolved)
		_CB_DIALOG_CACHE[key] = []
		return []
	try:
		with resolved.open("r", encoding="utf-8-sig") as handle:
			items = json.load(handle)
	except Exception as exc:
		logger.warning("Failed to load CB dataset from %s: %s", resolved, exc)
		_CB_DIALOG_CACHE[key] = []
		return []

	dialogs: List[Tuple[str, dict]] = []
	for idx, record in enumerate(items):
		if not isinstance(record, dict):
			continue
		dialog_id = record.get("uuid") or f"cb_{idx:05d}"
		dialogs.append((dialog_id, record))
	_CB_DIALOG_CACHE[key] = dialogs
	return dialogs


def seed_with_cb_anchor(
	dataset_path: Optional[Path],
	state: DialogSession,
	game: NegotiationGame,
	conversation: List[dict],
	max_pairs: int = 1,
) -> int:
	dialogs = load_cb_dialogs(dataset_path)
	if not dialogs:
		return 0
	resolved = dataset_path.resolve() if dataset_path else DEFAULT_CB_DATASET.resolve()
	key = str(resolved)
	total = len(dialogs)
	start_idx = _CB_ANCHOR_POINTERS.get(key, 0) % total

	for offset in range(total):
		dialog_idx = (start_idx + offset) % total
		dialog_id, record = dialogs[dialog_idx]
		pairs = _paired_segments(_segment_cb_dialog(record))
		if not pairs:
			continue
		seeded = 0
		for pair in pairs[:max_pairs]:
			(sys_role, sys_intent, sys_text), (usr_role, usr_intent, usr_text) = pair
			sys_da = NegotiationGame.map_craigslist_intent(sys_intent, NegotiationGame.SYS)
			usr_da = NegotiationGame.map_craigslist_intent(usr_intent, NegotiationGame.USR)
			state.add_single(game.SYS, sys_da, sys_text)
			state.add_single(game.USR, usr_da, usr_text)
			conversation.append(
				{
					"turn": len(conversation),
					"action_index": None,
					"system_dialog_act": sys_da,
					"system_utterance": sys_text,
					"user_selected_act": None,
					"user_dialog_act": usr_da,
					"user_utterance": usr_text,
					"turn_type": "anchor",
					"anchor_dialog_id": dialog_id,
				}
			)
			setattr(state, "_cb_scenario", record.get("scenario") or getattr(state, "_cb_scenario", None))
			seeded += 1
		if seeded > 0:
			setattr(state, "_anchor_dialog_id", dialog_id)
			setattr(state, "_last_cb_scenario", record.get("scenario"))
			setattr(state, "_cb_scenario", record.get("scenario"))
			_CB_ANCHOR_POINTERS[key] = (dialog_idx + 1) % total
			return seeded

	logger.warning("Unable to sample anchor turns from CB dataset; continuing without anchors.")
	_CB_ANCHOR_POINTERS[key] = (start_idx + 1) % total
	return 0


def _build_agents_and_game(args):
	backbone_model, *_ = create_factor_llm(args)
	ontology = NegotiationGame.get_game_ontology()
	sys_das = ontology["system"]["dialog_acts"]
	usr_das = ontology["user"]["dialog_acts"]

	seller = SellerModel(
		dialog_acts=sys_das,
		backbone_model=backbone_model,
		max_hist_num_turns=2,
		conv_examples=[],
		inference_args={
			"max_new_tokens": 96,
			"temperature": 0.9,
			"do_sample": True,
			"return_full_text": False,
		},
	)

	buyer = BuyerModel(
		dialog_acts=usr_das,
		backbone_model=backbone_model,
		max_hist_num_turns=2,
		conv_examples=[],
		inference_args={
			"max_new_tokens": 96,
			"temperature": 0.9,
			"do_sample": True,
			"return_full_text": False,
		},
	)

	planner = CBSystemPlanner(
		dialog_acts=sys_das,
		max_hist_num_turns=2,
		user_dialog_acts=usr_das,
		user_max_hist_num_turns=2,
		generation_model=backbone_model,
		conv_examples=[],
	)

	game = NegotiationGame(system_agent=seller, user_agent=buyer, max_conv_turns=args.max_turns)
	return backbone_model, planner, game, sys_das


def simulate_dialog(
	game: NegotiationGame,
	planner: CBSystemPlanner,
	mcts_cfg: dotdict,
	num_mcts_sims: int,
	max_turns: int,
	collect_preferences: bool,
	dialog_id: Optional[str],
	anchor_dataset: Optional[Path],
	max_anchor_pairs: int,
	persona_enabled: bool,
	user_mode: str,
	classify_user_act: bool,
	buyer_planner: Optional[CBBuyerPlanner],
) -> tuple[dict, List[dict]]:
	state = game.init_dialog()
	conversation: List[dict] = []
	preference_candidates: List[dict] = []

	seeded_pairs = 0
	if anchor_dataset is not None:
		seeded_pairs = seed_with_cb_anchor(
			anchor_dataset,
			state=state,
			game=game,
			conversation=conversation,
			max_pairs=max_anchor_pairs,
		)
	active_dialog_id = getattr(state, "_anchor_dialog_id", None) or dialog_id
	remaining_turns = max_turns if seeded_pairs == 0 else max(0, max_turns - seeded_pairs)

	persona_profile: Optional[dict] = None
	if persona_enabled:
		get_persona_fn = getattr(game.user_agent, "_get_persona_profile", None)
		if callable(get_persona_fn):
			persona_profile = get_persona_fn(state)

	for _ in range(remaining_turns):
		final_outcome = game.get_dialog_ended(state)
		if final_outcome != 0.0:
			break

		dialog_planner = OpenLoopMCTSParallel(game, planner, mcts_cfg)
		for _ in range(num_mcts_sims):
			dialog_planner.search(state)

		action_prob = dialog_planner.get_action_prob(state)
		state_rep = dialog_planner._to_string_rep(state)
		valid_moves = dialog_planner.valid_moves.get(state_rep)
		if valid_moves is None or np.sum(action_prob) == 0:
			continue
		best_action = int(np.argmax(action_prob))
		sys_da = game.system_agent.dialog_acts[best_action]
		sys_utt = game.system_agent.get_utterance(state.copy(), best_action)
		state.add_single(game.SYS, sys_da, sys_utt)

		user_selected_act = None
		if user_mode in {"planner", "hybrid"} and buyer_planner is not None:
			user_selected_act = buyer_planner.select_action(state)
		user_da, user_utt = game.user_agent.get_utterance_w_da(
			state,
			action=user_selected_act,
		)
		if user_mode == "planner" and user_selected_act:
			user_da = user_selected_act
		if classify_user_act or user_mode == "hybrid":
			user_da = classify_buyer_act(user_utt, fallback=user_da)
		state.add_single(game.USR, user_da, user_utt)

		conversation.append(
			{
				"turn": len(conversation),
				"action_index": best_action,
				"system_dialog_act": sys_da,
				"system_utterance": sys_utt,
				"user_selected_act": None,
				"user_dialog_act": user_da,
				"user_utterance": user_utt,
				"turn_type": "simulated",
				"anchor_dialog_id": None,
			}
		)

		if collect_preferences:
			preference_pair = get_preference_pair(
				action_prob,
				state_rep,
				game.system_agent.dialog_acts,
				valid_moves,
				dialog_planner.realizations_Vs,
			)
			if preference_pair:
				anchor_dialog_id = getattr(state, "_anchor_dialog_id", None)
				preference_candidates.append(
					{
						"turn": len(conversation) - 1,
						"state": state.copy(),
						"preference_pair": preference_pair,
						"dialog_turn_id": f"{active_dialog_id}_turn{len(conversation) - 1}" if active_dialog_id else None,
						"anchor_dialog_id": anchor_dialog_id,
					}
				)

		if user_da == NegotiationGame.B_ACCEPT:
			break

	final_outcome = game.get_dialog_ended(state)
	sim_result = {
		"dialog_id": active_dialog_id,
		"turns": conversation,
		"outcome": final_outcome,
		"persona_profile": persona_profile,
	}
	if not collect_preferences or final_outcome != 1.0:
		return sim_result, []
	return sim_result, preference_candidates


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Simulate Craigslist Bargain negotiations and export preference pairs."
	)
	parser.add_argument(
		"--llm",
		type=str,
		default="gpt-3.5-turbo",
		choices=["code-davinci-002", "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "chatgpt", "claude-haiku-3.5", "claude-3-5-haiku-20241022"],
		help="Backbone model identifier.",
	)
	parser.add_argument(
		"--gen-sentences",
		type=int,
		default=-1,
		help="Number of sentences for chat-based models (passed to chat wrappers).",
	)
	parser.add_argument(
		"--local-model-path",
		type=str,
		default="",
		help="Path to local HF model when using --llm local/gpt2.",
	)
	parser.add_argument(
		"--local-base-model",
		type=str,
		default="",
		help="Base model identifier when loading a PEFT adapter checkpoint.",
	)
	parser.add_argument(
		"--local-trust-remote-code",
		action="store_true",
		help="Allow executing remote code while loading local HF models.",
	)
	parser.add_argument(
		"--num-dialogs",
		type=int,
		default=5,
		help="Number of simulations to run.",
	)
	parser.add_argument(
		"--num-mcts-sims",
		type=int,
		default=20,
		help="Number of MCTS simulations per dialog turn.",
	)
	parser.add_argument(
		"--max-realizations",
		type=int,
		default=5,
		help="Maximum realizations tracked per state for OpenLoopMCTS.",
	)
	parser.add_argument(
		"--Q_0",
		type=float,
		default=0.25,
		help="Initial Q-value for unexplored actions.",
	)
	parser.add_argument(
		"--max-turns",
		type=int,
		default=6,
		help="Maximum dialog turns before forcing termination.",
	)
	parser.add_argument(
		"--user-mode",
		type=str,
		choices=["llm", "planner", "hybrid"],
		default="llm",
		help="Buyer strategy: 'llm' for free-form replies, 'planner' to follow heuristic acts, 'hybrid' to guide acts then classify.",
	)
	parser.add_argument(
		"--classify-user-act",
		action="store_true",
		help="Enable lightweight intent classification of buyer utterances.",
	)
	parser.add_argument(
		"--persona",
		action="store_true",
		help="Attach a random buyer persona profile for each simulation.",
	)
	parser.add_argument(
		"--anchor-dataset",
		type=str,
		default=str(DEFAULT_CB_DATASET),
		help="Path to a Craigslist Bargains JSON dataset for seeding anchors. Pass an empty string to disable.",
	)
	parser.add_argument(
		"--anchor-turns",
		type=int,
		default=2,
		help="Maximum number of anchor turns (seller+buyer pairs) to preload before simulation.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducibility.",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
		help="Logging verbosity.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional path to save raw simulation transcripts (JSONL).",
	)
	parser.add_argument(
		"--preference-output",
		type=Path,
		default=None,
		help="Optional path to export preference pairs (JSONL).",
	)
	return parser.parse_args()


def configure_logging(level: str) -> None:
	logging.basicConfig(
		level=getattr(logging, level),
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	)


def _resolve_anchor_path(anchor_arg: Optional[str]) -> Optional[Path]:
	if anchor_arg is None:
		default_path = DEFAULT_CB_DATASET.resolve()
		if default_path.exists():
			return default_path
		logger.warning("Default CB dataset not found at %s; anchors disabled.", default_path)
		return None
	anchor_arg = anchor_arg.strip()
	if not anchor_arg:
		return None
	candidate = Path(anchor_arg)
	if not candidate.is_absolute():
		candidate = (PROJECT_ROOT / anchor_arg).resolve()
	if candidate.exists():
		return candidate
	logger.warning("Anchor dataset not found at %s; anchors disabled.", candidate)
	return None


def main() -> None:
	args = parse_args()
	set_determinitic_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	configure_logging(args.log_level)

	backbone_model, planner, game, sys_das = _build_agents_and_game(args)
	logger.info("System dialog acts: %s", sys_das)
	logger.info("User dialog acts: %s", game.user_agent.dialog_acts)
	buyer_planner = None
	if args.user_mode in {"planner", "hybrid"}:
		buyer_planner = CBBuyerPlanner(game.user_agent.dialog_acts, rng=random.Random(args.seed))

	mcts_cfg = dotdict(
		{
			"cpuct": 1.0,
			"num_MCTS_sims": args.num_mcts_sims,
			"Q_0": args.Q_0,
			"max_realizations": args.max_realizations,
		}
	)

	preference_output_path = args.preference_output.resolve() if args.preference_output else None
	preference_enabled = preference_output_path is not None
	if preference_output_path:
		preference_output_path.parent.mkdir(parents=True, exist_ok=True)

	anchor_dataset_path = _resolve_anchor_path(args.anchor_dataset)

	dialog_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
	total_preference_pairs = 0
	successful_dialogs = 0
	results = []

	for sim_id in range(args.num_dialogs):
		default_dialog_id = f"cb_sim_{dialog_prefix}_{sim_id:04d}"
		sim_result, pref_candidates = simulate_dialog(
			game=game,
			planner=planner,
			mcts_cfg=mcts_cfg,
			num_mcts_sims=args.num_mcts_sims,
			max_turns=args.max_turns,
			collect_preferences=preference_enabled,
			dialog_id=default_dialog_id,
			anchor_dataset=anchor_dataset_path,
			max_anchor_pairs=args.anchor_turns,
			persona_enabled=args.persona,
			user_mode=args.user_mode,
			classify_user_act=args.classify_user_act,
			buyer_planner=buyer_planner,
		)
		actual_dialog_id = sim_result.get("dialog_id") or default_dialog_id
		logger.info("=== Simulation %d (%s) ===", sim_id + 1, actual_dialog_id)
		results.append(sim_result)
		persona_profile = sim_result.get("persona_profile")
		if persona_profile:
			logger.info(
				"Persona summary | Big-Five: %s | Decision style: %s",
				persona_profile.get("big_five", "N/A"),
				persona_profile.get("decision_making_style", "N/A"),
			)
			logger.info("Persona description: %s", persona_profile.get("description", ""))
		for turn in sim_result["turns"]:
			logger.info(
				"[Turn %d][%s] SELLER(%s): %s",
				turn["turn"],
				turn.get("turn_type", "simulated"),
				turn["system_dialog_act"],
				turn["system_utterance"],
			)
			logger.info(
				"[Turn %d][%s] BUYER(%s): %s",
				turn["turn"],
				turn.get("turn_type", "simulated"),
				turn["user_dialog_act"],
				turn["user_utterance"],
			)
		logger.info("Simulation outcome: %s", sim_result["outcome"])

		if preference_enabled and sim_result["outcome"] == 1.0 and pref_candidates:
			for candidate in pref_candidates:
				anchor_base = candidate.get("anchor_dialog_id")
				if anchor_base:
					dialog_turn_id = f"{anchor_base}_turn{candidate['turn']}"
				else:
					dialog_turn_id = candidate.get("dialog_turn_id") or f"{actual_dialog_id}_turn{candidate['turn']}"
				export_preference_pair(
					dialog_id=dialog_turn_id,
					state=candidate["state"],
					preference_pair=candidate["preference_pair"],
					system_role=game.SYS,
					output_path=preference_output_path,
				)
				total_preference_pairs += 1
			successful_dialogs += 1
			logger.info(
				"Exported %d preference pairs for dialog %s.",
				len(pref_candidates),
				actual_dialog_id,
			)

	if preference_enabled:
		logger.info(
			"Preference export summary: %d pairs written across %d successful dialogs to %s.",
			total_preference_pairs,
			successful_dialogs,
			preference_output_path,
		)

	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		with args.output.open("w", encoding="utf-8") as handle:
			for item in results:
				handle.write(json.dumps(item, ensure_ascii=False) + "\n")
		logger.info("Saved transcripts to %s", args.output)


if __name__ == "__main__":
	main()
