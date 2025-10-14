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
from utils.utils import dotdict, set_determinitic_seed
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)


def simulate_dialog(
	game: PersuasionGame,
	planner,
	mcts_cfg: dotdict,
	num_mcts_sims: int,
	max_turns: int,
) -> dict:
	state = game.init_dialog()
	conversation: List[dict] = []

	default_sys_da = (
		PersuasionGame.S_Greeting
		if PersuasionGame.S_Greeting in game.system_agent.dialog_acts
		else game.system_agent.dialog_acts[0]
	)
	default_usr_da = (
		PersuasionGame.U_Neutral
		if PersuasionGame.U_Neutral in game.user_agent.dialog_acts
		else game.user_agent.dialog_acts[0]
	)
	default_sys_utt = "Hello there! How are you doing today?"
	default_usr_utt = "I'm doing well, thanks. What's this about?"
	state.add_single(PersuasionGame.SYS, default_sys_da, default_sys_utt)
	state.add_single(PersuasionGame.USR, default_usr_da, default_usr_utt)
	conversation.append(
		{
			"turn": 0,
			"action_index": None,
			"system_dialog_act": default_sys_da,
			"system_utterance": default_sys_utt,
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

		next_state = game.get_next_state(state, best_action)
		sys_role, sys_da, sys_utt = next_state.history[-2]
		usr_role, usr_da, usr_utt = next_state.history[-1]

		conversation.append(
			{
				"turn": turn_idx + 1,
				"action_index": best_action,
				"system_dialog_act": sys_da,
				"system_utterance": sys_utt,
				"user_dialog_act": usr_da,
				"user_utterance": usr_utt,
			}
		)

		state = next_state

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
		default=1,
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
		default=3,
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

	backbone_model, SysModel, UsrModel, SysPlanner = create_factor_llm(args)
	ontology = PersuasionGame.get_game_ontology()
	exp_session = DialogSession(PersuasionGame.SYS, PersuasionGame.USR).from_history(EXP_DIALOG)

	system = SysModel(
		ontology["system"]["dialog_acts"],
		backbone_model,
		conv_examples=[exp_session],
		inference_args={
			"max_new_tokens": 128,
			"temperature": 0.0,
			"do_sample": False,
			"return_full_text": False,
		},
	)
	user = UsrModel(
		ontology["user"]["dialog_acts"],
		inference_args={
			"max_new_tokens": 128,
			"temperature": 0.6,
			"do_sample": True,
			"return_full_text": False,
		},
		backbone_model=backbone_model,
		conv_examples=[exp_session],
	)
	planner = SysPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[],
	)
	game = PersuasionGame(system, user)

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
