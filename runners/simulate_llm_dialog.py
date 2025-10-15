#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np

from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.model_factory import create_factor_llm
from core.helpers import DialogSession
from core.PersuadeePlanner import PersuadeeHeuristicPlanner
from utils.utils import dotdict, set_determinitic_seed
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
    persuader = SysModel(
        dialog_acts=sys_das,
        backbone_model=backbone_model,
        max_hist_num_turns=2,
        conv_examples=[],
        inference_args={
            "max_new_tokens": 128,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
        },
    )
    persuadee = UsrModel(dialog_acts=usr_das, backbone_model=backbone_model,
                         max_hist_num_turns=2, conv_examples=[],
                         inference_args={"max_new_tokens": 64, "temperature": 0.0})

    # Planner (policy & value/heuristic)
    planner = SysPlanner(dialog_acts=sys_das, max_hist_num_turns=2,
                         user_dialog_acts=usr_das, user_max_hist_num_turns=2,
                         generation_model=backbone_model, conv_examples=[])
    
    persuadee_planner = None
    if args.user_mode in {"planner", "hybrid"}:
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

	# seed with a default opening exchange so heuristics expecting a user turn work
	if len(state.history) == 0:
		default_sys_da = (
			PersuasionGame.S_Greeting
			if PersuasionGame.S_Greeting in game.system_agent.dialog_acts
			else game.system_agent.dialog_acts[0]
		)
		default_sys_utt = "Hello there! How are you doing today?"
		default_usr_da = (
			PersuasionGame.U_Neutral
			if PersuasionGame.U_Neutral in game.user_agent.dialog_acts
			else game.user_agent.dialog_acts[0]
		)
		default_usr_utt = "I'm doing well. What's this about?"
		state.add_single(PersuasionGame.SYS, default_sys_da, default_sys_utt)
		state.add_single(PersuasionGame.USR, default_usr_da, default_usr_utt)
		conversation.append(
			{
				"turn": 0,
				"action_index": None,
				"system_dialog_act": default_sys_da,
				"system_utterance": default_sys_utt,
				"user_selected_act": None,
				"user_dialog_act": default_usr_da,
				"user_utterance": default_usr_utt,
			}
		)

	for turn_idx in range(max_turns):
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
				"turn": turn_idx + 1,
				"action_index": best_action,
				"system_dialog_act": sys_da,
				"system_utterance": sys_utt,
				"user_selected_act": user_selected_act,
				"user_dialog_act": user_da,
				"user_utterance": user_utt,
			}
		)

		if user_da == PersuasionGame.U_Donate:
			break

	final_outcome = game.get_dialog_ended(state)
	return {
		"turns": conversation,
		"outcome": final_outcome,
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
		default=10,
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
		"--classify-user-act",
		action="store_true",
		help="Run an auxiliary classification step to assign persuadee dialog acts.",
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

	_, planner, persuadee_planner, game, sys_das = _build_agents_and_game(args)
	# logger.info("Using backbone model: %s", backbone_model.model_name)
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
		for turn in sim_result["turns"]:
			logger.info(
				"[Turn %d] SYS(%s): %s",
				turn["turn"],
				turn["system_dialog_act"],
				turn["system_utterance"],
			)
			logger.info(
				"[Turn %d] USR(%s): %s",
				turn["turn"],
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
