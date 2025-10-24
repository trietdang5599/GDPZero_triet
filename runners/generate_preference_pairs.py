#!/usr/bin/env python3

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
from utils.utils import (
	dotdict,
	export_preference_pair,
	get_preference_pair,
	seed_with_p4g_anchor,
	set_determinitic_seed,
)
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)


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
		inference_args={"max_new_tokens": 64, "temperature": 0.7},
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
	collect_preferences: bool = False,
	dialog_id: Optional[str] = None,
	persona_enabled: bool = False,
) -> tuple[dict, List[dict]]:
	state = game.init_dialog()
	conversation: List[dict] = []
	preference_candidates: List[dict] = []

	seeded_pairs = seed_with_p4g_anchor(state, game, conversation)
	remaining_turns = max_turns if seeded_pairs == 0 else max(0, max_turns - seeded_pairs)

	persona_profile: Optional[dict] = None
	if persona_enabled:
		get_persona_fn = getattr(game.user_agent, "_get_persona_profile", None)
		if callable(get_persona_fn):
			try:
				persona_profile = get_persona_fn(state)
			except TypeError:
				persona_profile = None
	if persona_enabled and persona_profile:
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
		state_rep = dialog_planner._to_string_rep(state)
		valid_moves = dialog_planner.valid_moves.get(state_rep)
		if valid_moves is None or np.sum(action_prob) == 0:
			continue
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

		if collect_preferences:
			preference_pair = get_preference_pair(
				action_prob,
				state_rep,
				game.system_agent.dialog_acts,
				valid_moves,
				dialog_planner.realizations_Vs,
			)
			if preference_pair:
				preference_candidates.append(
					{
						"turn": len(conversation) - 1,
						"state": state.copy(),
						"preference_pair": preference_pair,
						"dialog_turn_id": f"{dialog_id}_turn{len(conversation) - 1}" if dialog_id else None,
					}
				)

		if user_da == PersuasionGame.U_Donate:
			break

	final_outcome = game.get_dialog_ended(state)
	sim_result = {
		"dialog_id": dialog_id,
		"turns": conversation,
		"outcome": final_outcome,
		"persona_profile": persona_profile,
	}
	if not collect_preferences or final_outcome != 1.0:
		return sim_result, []
	return sim_result, preference_candidates


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Simulate a persuasion dialog where both agents are powered by LLMs."
	)
	parser.add_argument(
		"--llm",
		type=str,
		choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt", "claude-3-5-haiku-20241022"],
		default="gpt-3.5-turbo",
		help="Backbone model identifier (API-backed models only).",
	)
	parser.add_argument(
		"--gen-sentences",
		type=int,
		default=-1,
		help="Number of sentences for chat-based models (passed to OpenAI/Azure chat wrappers).",
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
	parser.add_argument(
		"--preference-output",
		type=Path,
		default=None,
		help="Optional path to export preference pairs (JSONL). Pairs are saved only for successful dialogs.",
	)
	parser.add_argument(
		"--persona",
		action="store_true",
		help="Enable persona generation for the Persuadee model before simulation.",
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

	preference_output_path = args.preference_output.resolve() if args.preference_output else None
	preference_enabled = preference_output_path is not None
	if preference_output_path:
		preference_output_path.parent.mkdir(parents=True, exist_ok=True)

	dialog_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
	total_preference_pairs = 0
	successful_preference_dialogs = 0

	results = []
	for sim_id in range(args.num_dialogs):
		dialog_id = f"sim_{dialog_prefix}_{sim_id:04d}"
		logger.info("=== Simulation %d (%s) ===", sim_id + 1, dialog_id)
		sim_result, pref_candidates = simulate_dialog(
			game,
			planner,
			mcts_cfg,
			args.num_mcts_sims,
			args.max_turns,
			user_mode=args.user_mode,
			classify_user_act=args.classify_user_act,
			user_planner=persuadee_planner,
			collect_preferences=preference_enabled,
			dialog_id=dialog_id,
			persona_enabled=args.persona,
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

		if preference_enabled and sim_result["outcome"] == 1.0 and pref_candidates:
			for candidate in pref_candidates:
				dialog_turn_id = candidate.get("dialog_turn_id") or f"{dialog_id}_turn{candidate['turn']}"
				export_preference_pair(
					dialog_id=dialog_turn_id,
					state=candidate["state"],
					preference_pair=candidate["preference_pair"],
					system_role=game.SYS,
					output_path=preference_output_path,
				)
				total_preference_pairs += 1
			successful_preference_dialogs += 1
			logger.info(
				"Exported %d preference pairs for dialog %s (outcome=%.1f).",
				len(pref_candidates),
				dialog_id,
				sim_result["outcome"],
			)

	if preference_enabled:
		logger.info(
			"Preference export summary: %d pairs written across %d successful dialogs to %s.",
			total_preference_pairs,
			successful_preference_dialogs,
			preference_output_path,
		)

	if args.output:
		import json

		args.output.parent.mkdir(parents=True, exist_ok=True)
		with args.output.open("w", encoding="utf-8") as f:
			for item in results:
				f.write(json.dumps(item, ensure_ascii=False) + "\n")
		logger.info("Saved transcripts to %s", args.output)


if __name__ == "__main__":
	main()
