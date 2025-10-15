#!/usr/bin/env python3

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from core.game import PersuasionGame
from core.gen_models import OpenAIModel
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from core.model_factory import create_factor_llm
from utils.prompt_examples import EXP_DIALOG
from utils.utils import dotdict, export_preference_pair, get_preference_pair

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _log_turn_details(
	enabled: bool,
	context: str,
	human_resp: str,
	human_da: str,
	mcts_resp: str,
	mcts_da: str,
) -> None:
	if not enabled:
		return
	logger.info("Context:\n%s", context)
	logger.info("human resp: %s", human_resp)
	logger.info("human da: %s", human_da)
	logger.info("mcts resp: %s", mcts_resp)
	logger.info("mcts da: %s", mcts_da)


def _init_models(cmd_args):
	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology["system"]["dialog_acts"]
	user_da = game_ontology["user"]["dialog_acts"]
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR

	exp_dialog = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	backbone_model, SysModel, UsrModel, SysPlanner = create_factor_llm(cmd_args)

	system = SysModel(
		sys_da,
		backbone_model,
		conv_examples=[exp_dialog],
		inference_args={
			"max_new_tokens": 80,
			"temperature": 0.7,
			"do_sample": True,
			"return_full_text": False,
		},
	)
	user = UsrModel(
		user_da,
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,
			"return_full_text": False,
		},
		backbone_model=backbone_model,
		conv_examples=[exp_dialog],
	)
	planner = SysPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_dialog],
	)
	game = PersuasionGame(system, user)
	return system, user, planner, game, backbone_model



def _map_system_da(raw_das, system_dialog_acts) -> str:
	sys_da = set(raw_das)
	intersected = sys_da.intersection(system_dialog_acts)
	if not intersected:
		return "other"
	return list(intersected)[-1]


def _load_dialogs(path: Path):
	with path.open("rb") as f:
		return pickle.load(f)


def generate_preferences(cmd_args):
	system, user, planner, game, backbone_model = _init_models(cmd_args)
	all_dialogs = _load_dialogs(PROJECT_ROOT / "data/p4g/300_dialog_turn_based.pkl")

	args = dotdict(
		{
			"cpuct": 1.0,
			"num_MCTS_sims": cmd_args.num_mcts_sims,
			"Q_0": cmd_args.Q_0,
			"max_realizations": cmd_args.max_realizations,
		}
	)

	output_path: Optional[Path] = Path(cmd_args.output).resolve() if cmd_args.output else None
	if output_path:
		output_path.parent.mkdir(parents=True, exist_ok=True)

	bad_dialogs = {
		"20180808-024552_152_live",
		"20180723-100140_767_live",
		"20180825-080802_964_live",
	}

	total_pairs = 0
	total_dialogs = 0
	pbar = tqdm(total=cmd_args.num_dialogs, desc="evaluating")
	for did in all_dialogs.keys():
		if did in bad_dialogs:
			logger.debug("Skipping dialog id: %s (known bad dialog)", did)
			continue
		if total_dialogs == cmd_args.num_dialogs:
			break

		dialog = all_dialogs[did]
		context = ""
		state = game.init_dialog()
		pending_pairs = []
		donation_success = False

		for turn_idx, turn in enumerate(dialog["dialog"]):
			if len(turn["ee"]) == 0:
				break
			if turn_idx == len(dialog["dialog"]) - 1:
				break

			usr_utt = " ".join(turn["ee"]).strip()
			raw_usr_da = dialog["label"][turn_idx]["ee"][-1]
			usr_da = PersuasionGame.map_user_da(raw_usr_da)

			sys_utt = " ".join(turn["er"]).strip()
			sys_da = _map_system_da(dialog["label"][turn_idx]["er"], system.dialog_acts)

			state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)

			context = f"""
			{context}
			Persuader: {sys_utt}
			Persuadee: {usr_utt}
			""".replace("\t", "").strip()

			if usr_da == PersuasionGame.U_Donate:
				logger.info(
					"[Preference] Dialog %s success at turn %s with response: %s",
					did,
					turn_idx,
					usr_utt,
				)
				donation_success = True
				break
			elif usr_da == PersuasionGame.U_NoDonation:
				logger.info(
					"[Preference] Dialog %s failure at turn %s with response: %s",
					did,
					turn_idx,
					usr_utt,
				)
				break

			if isinstance(backbone_model, OpenAIModel):
				backbone_model._cached_generate.cache_clear()

			dialog_planner = OpenLoopMCTS(game, planner, args)
			for _ in range(args.num_MCTS_sims):
				dialog_planner.search(state)

			probabilities = dialog_planner.get_action_prob(state)
			hashable_state = dialog_planner._to_string_rep(state)
			valid_moves = dialog_planner.valid_moves.get(hashable_state)
			if valid_moves is None or np.sum(probabilities) == 0:
				continue

			action_idx = int(np.argmax(probabilities))
			mcts_policy_next_da = system.dialog_acts[action_idx]

			# print detail logs
			if cmd_args.log_turn_details:
				mcts_pred_rep = dialog_planner.get_best_realization(state, action_idx)
				human_resp = " ".join(dialog["dialog"][turn_idx + 1]["er"]).strip()
				next_sys_das = set(dialog["label"][turn_idx + 1]["er"])
				next_sys_da = (
					list(next_sys_das.intersection(system.dialog_acts))[-1]
					if next_sys_das and next_sys_das.intersection(system.dialog_acts)
					else "other"
				)
				_log_turn_details(
					cmd_args.log_turn_details,
					context,
					human_resp,
					next_sys_da,
					mcts_pred_rep,
					mcts_policy_next_da,
				)

			realizations_vs = getattr(dialog_planner, "realizations_Vs", None)
			preference_pair = get_preference_pair(
				probabilities=probabilities,
				state_rep=hashable_state,
				dialog_acts=system.dialog_acts,
				valid_moves=valid_moves,
				realizations_vs=realizations_vs,
			)

			if preference_pair:
				pending_pairs.append(
					{
						"dialog_id": did,
						"state": state.copy(),
						"hashable_state": hashable_state,
						"preference_pair": preference_pair,
					}
				)
			
		should_export = donation_success or not cmd_args.only_success
		pbar.update(1)
		if should_export and pending_pairs:
			for pair in pending_pairs:
				export_preference_pair(
					dialog_id=pair["dialog_id"],
					state=pair["state"],
					preference_pair=pair["preference_pair"],
					system_role=game.SYS,
					output_path=output_path,
				)
				total_pairs += 1
			total_dialogs += 1
		print(f"Total dialogs with exported preferences: {total_dialogs}/{cmd_args.num_dialogs}, total pairs: {total_pairs}")
	logger.info("Preference generation complete: %s pairs written in success dialog %s.",
             total_pairs, total_dialogs)


def parse_args():
	parser = argparse.ArgumentParser(description="Generate preference pairs using GDPZero MCTS.")
	parser.add_argument(
		"--llm",
		type=str,
		default="local",
		choices=[
			"code-davinci-002",
			"chatgpt",
			"gpt-3.5-turbo",
			"gpt2",
			"qwen2.5-0.5b",
			"qwen2.5-7b",
			"llamda-3-8b",
			"deepseek-r1",
			"local",
		],
		help="Backbone model identifier.",
	)
	parser.add_argument(
		"--local-model-path",
		type=str,
		default="",
		help="Path to local Hugging Face model when using --llm local/gpt2.",
	)
	parser.add_argument(
		"--local-trust-remote-code",
		action="store_true",
		help="Allow executing remote code when loading a local Hugging Face model.",
	)
	parser.add_argument(
		"--num-dialogs",
		type=int,
		default=20,
		help="Number of dialogs to process.",
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
		help="Maximum realizations tracked per state in OpenLoopMCTS.",
	)
	parser.add_argument(
		"--Q_0",
		type=float,
		default=0.0,
		help="Initial Q-value baseline for unexplored actions.",
	)
	parser.add_argument(
		"--gen-sentences",
		type=int,
		default=-1,
		help="Number of sentences to generate for OpenAI chat models.",
	)
	parser.add_argument(
		"--only-success",
		action="store_true",
		help="Only export preference pairs for dialogs where donation succeeds.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="",
		help="Optional output path for the generated preference JSONL file.",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
		help="Logging level.",
	)
	parser.add_argument(
		"--log-turn-details",
		action="store_true",
		help="Log per-turn context and responses when generating preferences.",
	)
	return parser.parse_args()


def main():
	cmd_args = parse_args()

	log_dir = PROJECT_ROOT / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	log_path = log_dir / f"pref_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

	file_handler = logging.FileHandler(log_path, encoding="utf-8")
	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(getattr(logging, cmd_args.log_level.upper(), logging.INFO))
	logging.basicConfig(
		level=logging.DEBUG,
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
		handlers=[stream_handler, file_handler],
	)
	logger.info("Writing logs to %s", log_path)

	generate_preferences(cmd_args)


if __name__ == "__main__":
	main()
