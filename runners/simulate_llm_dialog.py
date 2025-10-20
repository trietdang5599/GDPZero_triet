#!/usr/bin/env python3

import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np

from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.model_factory import create_factor_llm
from core.helpers import DialogSession
from core.PersuadeePlanner import PersuadeeHeuristicPlanner, PersuadeeLLMPlanner
from utils.utils import dotdict, set_determinitic_seed
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
_P4G_DIALOG_CACHE: Optional[List[Tuple[str, dict]]] = None


def _load_p4g_dialogs() -> List[Tuple[str, dict]]:
	global _P4G_DIALOG_CACHE
	if _P4G_DIALOG_CACHE is not None:
		return _P4G_DIALOG_CACHE
	data_path = PROJECT_ROOT / "data" / "p4g" / "300_dialog_turn_based.pkl"
	if not data_path.exists():
		logger.warning("P4G dataset not found at %s – simulations will start empty.", data_path)
		_P4G_DIALOG_CACHE = []
		return _P4G_DIALOG_CACHE
	try:
		with data_path.open("rb") as handle:
			raw_data: Dict[str, dict] = pickle.load(handle)
	except Exception as exc:  # pragma: no cover
		logger.warning("Failed to load P4G dataset from %s: %s", data_path, exc)
		_P4G_DIALOG_CACHE = []
		return _P4G_DIALOG_CACHE
	items = list(raw_data.items())
	if not items:
		logger.warning("P4G dataset at %s is empty.", data_path)
	_P4G_DIALOG_CACHE = items
	return _P4G_DIALOG_CACHE


def _map_system_da(raw_das, system_dialog_acts: List[str]) -> str:
	if not raw_das:
		return system_dialog_acts[0]
	if isinstance(raw_das, (list, tuple, set)):
		candidates = [da for da in raw_das if da in system_dialog_acts]
		if candidates:
			return candidates[-1]
		if "other" in system_dialog_acts:
			return "other"
		return raw_das[-1] if raw_das and isinstance(raw_das[-1], str) else system_dialog_acts[0]
	if raw_das in system_dialog_acts:
		return raw_das
	return system_dialog_acts[0]


def _seed_with_p4g_anchor(state: DialogSession, game: PersuasionGame, conversation: List[dict]) -> int:
	dialog_items = _load_p4g_dialogs()
	if not dialog_items:
		return 0
	max_attempts = min(20, len(dialog_items))
	for _ in range(max_attempts):
		dialog_id, dialog_entry = random.choice(dialog_items)
		pairs_available = min(len(dialog_entry.get("dialog", [])), len(dialog_entry.get("label", [])))
		if pairs_available <= 0:
			continue
		num_pairs = 2
		seeded = 0
		for idx in range(num_pairs):
			turn = dialog_entry["dialog"][idx]
			if not turn.get("ee"):
				break
			sys_utt_raw = " ".join(turn.get("er", [])).strip()
			usr_utt_raw = " ".join(turn.get("ee", [])).strip()
			if not sys_utt_raw and not usr_utt_raw:
				continue
			label_entry = dialog_entry["label"][idx]
			sys_da = _map_system_da(label_entry.get("er"), game.system_agent.dialog_acts)
			raw_usr_labels = label_entry.get("ee") or []
			raw_usr_da = raw_usr_labels[-1] if raw_usr_labels else PersuasionGame.U_Neutral
			usr_da = PersuasionGame.map_user_da(raw_usr_da)
			sys_utt = sys_utt_raw or "..."
			usr_utt = usr_utt_raw or "..."
			state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)
			conversation.append(
				{
					"turn": len(conversation),
					"action_index": None,
					"system_dialog_act": sys_da,
					"system_utterance": sys_utt,
					"user_selected_act": None,
					"user_dialog_act": usr_da,
					"user_utterance": usr_utt,
					"turn_type": "anchor",
					"anchor_dialog_id": dialog_id,
				}
			)
			seeded += 1
			if idx == pairs_available - 1:
				break
		if seeded > 0:
			return seeded
	logger.warning("Unable to sample anchor turns from P4G dataset; proceeding without anchors.")
	return 0


def _build_agents_and_game(args):
	"""
	Dùng factory có sẵn của bạn để tạo backbone model + lớp chat.
	"""
	backbone_model, SysModel, UsrModel, SysPlanner = create_factor_llm(args)

	ontology = PersuasionGame.get_game_ontology()
	sys_das = ontology["system"]["dialog_acts"]
	usr_das = ontology["user"]["dialog_acts"]

	# Persuader / Persuadee models (NLG)
	# Enable sampling for Persuader so OpenLoop MCTS can cache
	# multiple realizations per action and produce preference pairs.
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR
	exp_dialog = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	persuader = SysModel(
		dialog_acts=sys_das,
		backbone_model=backbone_model,
		max_hist_num_turns=2,
		conv_examples=[exp_dialog],
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"do_sample": True,
			"return_full_text": False,
		},
	)
	persuadee = UsrModel(
		dialog_acts=usr_das,
		backbone_model=backbone_model,
		max_hist_num_turns=2,
		conv_examples=[exp_dialog],
		inference_args={"max_new_tokens": 64, "temperature": 0.5},
	)

	# Planner (policy & value/heuristic)
	planner = SysPlanner(
		dialog_acts=sys_das,
		max_hist_num_turns=2,
		user_dialog_acts=usr_das,
		user_max_hist_num_turns=2,
		generation_model=backbone_model,
		conv_examples=[exp_dialog],
	)

	persuadee_planner = None
	if args.user_mode in {"planner", "hybrid"}:
		if getattr(args, "user_planner", "heuristic") == "llm":
			persuadee_planner = PersuadeeLLMPlanner(
				dialog_acts=persuadee.dialog_acts,
				generation_model=backbone_model,
				max_hist_num_turns=2,
				seed=args.seed,
			)
		else:
			persuadee_planner = PersuadeeHeuristicPlanner(
				persuadee.dialog_acts,
				donate_prob=args.planner_donate_prob,
				seed=args.seed,
			)

	# Game
	game = PersuasionGame(system_agent=persuader, user_agent=persuadee, max_conv_turns=args.max_turns)
	return backbone_model, planner, persuadee_planner, game, sys_das


def simulate_dialog(
	game: PersuasionGame,
	planner,
	mcts_cfg: dotdict,
	num_mcts_sims: int,
	max_turns: int,
	user_mode: str,
	classify_user_act: bool,
	user_planner: PersuadeeHeuristicPlanner | None = None,
) -> dict:
	state = game.init_dialog()
	conversation: List[dict] = []

	seeded_pairs = _seed_with_p4g_anchor(state, game, conversation)
	remaining_turns = max_turns if seeded_pairs == 0 else max(0, max_turns - seeded_pairs)

	persona_profile: Optional[dict] = None
	get_persona_fn = getattr(game.user_agent, "_get_persona_profile", None)
	if callable(get_persona_fn):
		try:
			persona_profile = get_persona_fn(state)
		except TypeError:
			persona_profile = None
	if persona_profile:
		logger.info(
			"Persona profile | Big-Five: %s | Decision-Making: %s",
			persona_profile.get("big_five", "N/A"),
			persona_profile.get("decision_making_style", "N/A"),
		)
		logger.info("Persona description: %s", persona_profile.get("description", ""))

	for _ in range(remaining_turns):
		final_outcome = game.get_dialog_ended(state)
		if final_outcome != 0.0:
			break

		dialog_planner = OpenLoopMCTS(game, planner, mcts_cfg)
		for _ in range(num_mcts_sims):
			dialog_planner.search(state)

		action_prob = dialog_planner.get_action_prob(state)
		best_action = int(np.argmax(action_prob))
		sys_da = game.system_agent.dialog_acts[best_action]
		sys_utt = game.system_agent.get_utterance(state.copy(), best_action)
		state.add_single(PersuasionGame.SYS, sys_da, sys_utt)

		user_selected_act = None
		if user_mode in {"planner", "hybrid"} and user_planner is not None:
			user_selected_act = user_planner.select_action(state)

		user_da, user_utt = game.user_agent.get_utterance_w_da(
			state,
			action=user_selected_act,
			classify=classify_user_act or user_mode == "hybrid",
		)
		if user_mode in {"planner", "hybrid"} and user_selected_act and user_da == PersuasionGame.U_Neutral:
			user_da = user_selected_act
		state.add_single(PersuasionGame.USR, user_da, user_utt)

		conversation.append(
			{
				"turn": len(conversation),
				"action_index": best_action,
				"system_dialog_act": sys_da,
				"system_utterance": sys_utt,
				"user_selected_act": user_selected_act,
				"user_dialog_act": user_da,
				"user_utterance": user_utt,
				"turn_type": "simulated",
				"anchor_dialog_id": None,
			}
		)

		if user_da == PersuasionGame.U_Donate:
			break

	final_outcome = game.get_dialog_ended(state)
	return {
		"turns": conversation,
		"outcome": final_outcome,
		"persona_profile": persona_profile,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Simulate a persuasion dialog where both agents are powered by LLMs."
	)
	parser.add_argument(
		"--llm",
		type=str,
		default="qwen2.5-0.5b",
		help="Backbone model identifier (same choices as runners/gdpzero).",
	)
	parser.add_argument(
		"--gen-sentences",
		type=int,
		default=-1,
		help="Number of sentences for chat-based models (passed to OpenAI/Azure chat wrappers).",
	)
	parser.add_argument(
		"--local-model-path",
		type=str,
		default="",
		help="Path to local HF model when using --llm local/gpt2.",
	)
	parser.add_argument(
		"--local-trust-remote-code",
		action="store_true",
		help="Allow executing remote code when loading local HF model.",
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
		help="Number of MCTS simulations per turn.",
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
		default=5,
		help="Maximum dialog turns before forcing termination.",
	)
	parser.add_argument(
		"--user-mode",
		type=str,
		choices=["llm", "planner", "hybrid"],
		default="llm",
		help="Strategy for Persuadee dialog acts: 'llm' for free-form, 'planner' for heuristic acts, 'hybrid' for planner hint plus classification.",
	)
	parser.add_argument(
		"--user-planner",
		type=str,
		choices=["heuristic", "llm"],
		default="heuristic",
		help=(
			"When --user-mode is 'planner' or 'hybrid': choose 'heuristic' (mapping + randomness) or 'llm' "
			"(content-aware action selection from last 1–2 Persuader utterances)."
		),
	)
	parser.add_argument(
		"--classify-user-act",
		action="store_true",
		help="Run an auxiliary classification step to assign persuadee dialog acts. Dùng prompt để LLM phân loại hành động của persuadee.",
	)
	parser.add_argument(
		"--planner-donate-prob",
		type=float,
		default=0.4,
		help="Base probability for the heuristic planner to select donate when faced with a donation proposition.",
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
		help="Logging level for terminal output.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional path to save simulation transcripts (JSONL).",
	)
	return parser.parse_args()


def configure_logging(level: str) -> None:
	logging.basicConfig(
		level=getattr(logging, level),
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	)


def main() -> None:
	args = parse_args()
	configure_logging(args.log_level)
	set_determinitic_seed(args.seed)
	random.seed(args.seed)

	_, planner, persuadee_planner, game, sys_das = _build_agents_and_game(args)
	logger.info("System dialog acts: %s", sys_das)

	mcts_cfg = dotdict(
		{
			"cpuct": 1.0,
			"num_MCTS_sims": args.num_mcts_sims,
			"Q_0": args.Q_0,
			"max_realizations": args.max_realizations,
		}
	)

	results = []
	for sim_id in range(args.num_dialogs):
		logger.info("=== Simulation %d ===", sim_id + 1)
		sim_result = simulate_dialog(
			game,
			planner,
			mcts_cfg,
			args.num_mcts_sims,
			args.max_turns,
			user_mode=args.user_mode,
			classify_user_act=args.classify_user_act,
			user_planner=persuadee_planner,
		)
		results.append(sim_result)
		pp = sim_result.get("persona_profile") or {}
		if pp:
			logger.info(
				"Persona summary | Big-Five: %s | Decision-Making: %s",
				pp.get("big_five", "N/A"),
				pp.get("decision_making_style", "N/A"),
			)
			logger.info("Persona description: %s", pp.get("description", ""))
		for turn in sim_result["turns"]:
			logger.info(
				"[Turn %d][%s] SYS(%s): %s",
				turn["turn"],
				turn.get("turn_type", "simulated"),
				turn["system_dialog_act"],
				turn["system_utterance"],
			)
			logger.info(
				"[Turn %d][%s] USR(%s): %s",
				turn["turn"],
				turn.get("turn_type", "simulated"),
				turn["user_dialog_act"],
				turn["user_utterance"],
			)
		logger.info("Simulation outcome: %s", sim_result["outcome"])

	if args.output:
		import json

		args.output.parent.mkdir(parents=True, exist_ok=True)
		with args.output.open("w", encoding="utf-8") as f:
			for item in results:
				f.write(json.dumps(item, ensure_ascii=False) + "\n")
		logger.info("Saved transcripts to %s", args.output)


if __name__ == "__main__":
	main()
