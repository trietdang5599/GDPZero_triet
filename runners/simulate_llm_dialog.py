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

DEFAULT_ANCHOR_DATASET = (PROJECT_ROOT / "data" / "p4g" / "300_dialog_turn_based-test.jsonl").resolve()

from core.game import PersuasionGame
from core.model_factory import create_factor_llm
from core.gen_models import LocalModel, OpenAIChatModel, AzureOpenAIChatModel
from core.helpers import DialogSession
from core.PersuadeePlanner import PersuadeeLLMPlanner
from utils.utils import (
	seed_with_p4g_anchor,
	set_determinitic_seed,
)
from utils.prompt_examples import EXP_DIALOG
import nltk
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass


logger = logging.getLogger(__name__)


def _select_persuadee_backbone(args, default_backbone):
	api_llm = getattr(args, "persuadee_api_llm", "")
	model_name = getattr(args, "persuadee_model_name", "")
	if api_llm:
		if model_name:
			logger.warning(
				"Ignoring --persuadee-model-name because --persuadee-api-llm is provided for Persuadee."
			)
		api_provider = getattr(args, "persuadee_api_provider", "openai")
		if api_provider == "azure":
			logger.info("Using Azure OpenAI chat model for Persuadee: %s", api_llm)
			return AzureOpenAIChatModel(api_llm, args.gen_sentences)
		logger.info("Using OpenAI chat model for Persuadee: %s", api_llm)
		return OpenAIChatModel(api_llm, args.gen_sentences)
	if model_name and model_name != getattr(args, "llm", ""):
		logger.info("Loading Persuadee model from Hugging Face: %s", model_name)
		return LocalModel(model_name, trust_remote_code=True)
	return default_backbone


def _select_persuader_backbone(args, default_backbone):
	model_path = getattr(args, "persuader_model_path", "")
	if model_path:
		logger.info("Loading persuader checkpoint: %s", model_path)
		model_kwargs = {}
		base_model = getattr(args, "persuader_base_model", "")
		persuader_model_name = getattr(args, "persuader_model_name", "")
		persuadee_model_name = getattr(args, "persuadee_model_name", "")
		if base_model:
			model_kwargs["base_model_name_or_path"] = base_model
		elif persuader_model_name:
			model_kwargs["base_model_name_or_path"] = persuader_model_name
		elif persuadee_model_name:
			model_kwargs["base_model_name_or_path"] = persuadee_model_name
		else:
			model_kwargs["base_model_name_or_path"] = getattr(args, "llm", "")
		return LocalModel(
			model_path,
			trust_remote_code=True,
			model_kwargs=model_kwargs or None,
		)
	persuader_model_name = getattr(args, "persuader_model_name", "")
	if persuader_model_name:
		logger.info("Loading persuader model from Hugging Face: %s", persuader_model_name)
		return LocalModel(persuader_model_name, trust_remote_code=True)
	return default_backbone


def _build_agents_and_game(args):

	backbone_model, SysModel, UsrModel, SysPlanner = create_factor_llm(args)

	ontology = PersuasionGame.get_game_ontology()
	sys_das = ontology["system"]["dialog_acts"]
	usr_das = ontology["user"]["dialog_acts"]

	# Persuader / Persuadee models (NLG)
	# Enable sampling so the Persuader can explore diverse follow-ups when generating utterances.
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR
	exp_dialog = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	persuadee_backbone = _select_persuadee_backbone(args, default_backbone=backbone_model)
	persuader_backbone = _select_persuader_backbone(args, default_backbone=backbone_model)

	persuader = SysModel(
		dialog_acts=sys_das,
		backbone_model=persuader_backbone,
		max_hist_num_turns=3,
		conv_examples=[exp_dialog],
		inference_args={
			"max_new_tokens": 256,
			"temperature": 0.7,
			"top_p": 0.8,
			"do_sample": True,
			"return_full_text": False,
		},
		use_persona_context=getattr(args, "persuader_use_persona", False),
	)
	persuadee = UsrModel(
		dialog_acts=usr_das,
		backbone_model=persuadee_backbone,
		max_hist_num_turns=1,
		conv_examples=[exp_dialog],
		inference_args={"max_new_tokens": 128, "temperature": 0.2, "repetition_penalty": 1.0, "return_full_text": False},
	)

	# Planner (policy) for Persuader uses an LLM-only classifier over dialog acts.
	# planner = PersuaderLLMPlanner(
	# 	dialog_acts=sys_das,
	# 	generation_model=persuader_backbone,
	# 	max_hist_num_turns=2,
	# 	seed=args.seed,
	# )

	planner = SysPlanner(
		dialog_acts=sys_das,
		max_hist_num_turns=2,
		user_dialog_acts=usr_das,
		user_max_hist_num_turns=2,
		generation_model=persuader_backbone,
		conv_examples=[exp_dialog],
	)

	persuadee_planner = PersuadeeLLMPlanner(
		dialog_acts=persuadee.dialog_acts,
		generation_model=persuadee_backbone,
		max_hist_num_turns=2,
		seed=args.seed,
	)

	# Game
	game = PersuasionGame(system_agent=persuader, user_agent=persuadee, max_conv_turns=args.max_turns)
	return persuader_backbone, planner, persuadee_planner, game, sys_das


def simulate_dialog(
	game: PersuasionGame,
	planner,
	max_turns: int,
	classify_user_act: bool,
	user_planner: PersuadeeLLMPlanner | None = None,
	dialog_id: Optional[str] = None,
	anchor_dataset: Optional[Path] = None,
	persuadee_persona_enabled: bool = False,
) -> dict:
	state = game.init_dialog()
	conversation: List[dict] = []
	if anchor_dataset is not None:
		seeded_pairs = seed_with_p4g_anchor(
			dataset_path=anchor_dataset,
			state=state,
			game=game,
			conversation=conversation,
		)
	else:
		seeded_pairs = 0
	remaining_turns = max_turns if seeded_pairs == 0 else max(0, max_turns - seeded_pairs)
	
	persona_profile: Optional[dict] = None
	if persuadee_persona_enabled:
		get_persona_fn = getattr(game.user_agent, "_get_persona_profile", None)
		if callable(get_persona_fn):
			try:
				persona_profile = get_persona_fn(state)
			except TypeError:
				persona_profile = None
	if persona_profile and persuadee_persona_enabled:
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

		valid_moves = planner.get_valid_moves(state)
		action_prob, _ = planner.predict(state)
		action_prob = np.asarray(action_prob, dtype=np.float64)
		valid_moves = np.asarray(valid_moves, dtype=np.float64)
		action_prob *= valid_moves

		if np.sum(action_prob) <= 0.0:
			valid_indices = np.flatnonzero(valid_moves)
			if valid_indices.size == 0:
				logger.debug("No valid moves available for dialog %s; terminating early.", dialog_id or "N/A")
				break
			best_action = int(np.random.choice(valid_indices))
		else:
			action_prob /= action_prob.sum()
			best_action = int(np.random.choice(len(action_prob), p=action_prob))
		sys_da = game.system_agent.dialog_acts[best_action]
		sys_utt = game.system_agent.get_utterance(state.copy(), best_action)
		state.add_single(PersuasionGame.SYS, sys_da, sys_utt)

		user_selected_act = user_planner.select_action(state) if user_planner is not None else None

		user_da, user_utt = game.user_agent.get_utterance_w_da(
			state,
			action=user_selected_act,
			classify=classify_user_act,
		)
		if user_selected_act and user_da == PersuasionGame.U_Neutral:
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
	sim_result = {
		"dialog_id": dialog_id,
		"turns": conversation,
		"outcome": final_outcome,
		"persona_profile": persona_profile,
	}
	return sim_result


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Simulate a persuasion dialog where both agents are powered by LLMs."
	)
	parser.add_argument(
		"--llm",
		type=str,
		default="local",
		help="Backbone model identifier (same choices as runners/gdpzero).",
	)
	parser.add_argument(
		"--gen-sentences",
		type=int,
		default=3,
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
		"--persuadee-model-name",
		type=str,
		default="",
		help="Optional Hugging Face model identifier for the Persuadee agent (when different from --llm).",
	)
	parser.add_argument(
		"--persuadee-api-llm",
		type=str,
		default="",
		help="Optional API-based chat model identifier for the Persuadee (e.g., gpt-4o-mini).",
	)
	parser.add_argument(
		"--persuadee-api-provider",
		type=str,
		choices=["openai", "azure"],
		default="openai",
		help="API provider to use when --persuadee-api-llm is set.",
	)
	parser.add_argument(
		"--persuader-model-path",
		type=str,
		default="",
		help="Checkpoint directory for the Persuader agent (e.g., DPO fine-tuned model).",
	)
	parser.add_argument(
		"--persuader-base-model",
		type=str,
		default="",
		help="Base model identifier used when loading a Persuader adapter checkpoint.",
	)
	parser.add_argument(
		"--persuader-model-name",
		type=str,
		default="",
		help="Hugging Face model identifier for the Persuader agent (used if --persuader-model-path is not provided).",
	)
	parser.add_argument(
		"--num-dialogs",
		type=int,
		default=5,
		help="Number of simulations to run.",
	)
	parser.add_argument(
		"--max-turns",
		type=int,
		default=8,
		help="Maximum dialog turns before forcing termination.",
	)
	parser.add_argument(
		"--classify-user-act",
		action="store_true",
		help="Run an auxiliary classification step to assign persuadee dialog acts.",
	)
	parser.add_argument(
		"--use-persona",
		action="store_true",
		help="Expose persuadee personality and decision-making style to Persuadee prompts.",
	)
	parser.add_argument(
		"--persuader-use-persona",
		action="store_true",
		help="Expose persuadee personality and decision-making style to Persuader prompts.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for reproducibility. Omit to allow fully stochastic runs.",
	)
	parser.add_argument(
		"--cuda-deterministic",
		action="store_true",
		help="Force deterministic CUDA kernels (may reduce sampling diversity).",
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
		"--dataset-path",
		type=Path,
		default=DEFAULT_ANCHOR_DATASET,
		help="Path to the dialog dataset used for seeding conversations (default: P4G test split).",
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
	set_determinitic_seed(args.seed, enforce_determinism=args.cuda_deterministic)
	if args.seed is not None:
		random.seed(args.seed)

	_, planner, persuadee_planner, game, sys_das = _build_agents_and_game(args)
	logger.info("System dialog acts: %s", sys_das)

	anchor_dataset = None
	if args.dataset_path:
		anchor_dataset = Path(args.dataset_path).expanduser()
		if not anchor_dataset.is_absolute():
			anchor_dataset = (PROJECT_ROOT / anchor_dataset).resolve()
		else:
			anchor_dataset = anchor_dataset.resolve()
		if not anchor_dataset.exists():
			logger.warning("Anchor dataset %s does not exist; seeding will be skipped.", anchor_dataset)
			anchor_dataset = None
		else:
			logger.info("Using anchor dataset at %s", anchor_dataset)

	dialog_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

	results = []
	for sim_id in range(args.num_dialogs):
		dialog_id = f"sim_{dialog_prefix}_{sim_id:04d}"
		logger.info("=== Simulation %d (%s) ===", sim_id + 1, dialog_id)
		sim_result = simulate_dialog(
			game,
			planner,
			args.max_turns,
			classify_user_act=args.classify_user_act,
			user_planner=persuadee_planner,
			dialog_id=dialog_id,
			anchor_dataset=anchor_dataset,
			persuadee_persona_enabled=args.use_persona,
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
