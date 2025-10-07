import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import math
import multiprocessing as mp
import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np

from tqdm.auto import tqdm
from core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.players import (
	PersuadeeModel, PersuaderModel, P4GSystemPlanner,
	PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


BAD_DIALOG_IDS = {
	'20180808-024552_152_live',
	'20180723-100140_767_live',
	'20180825-080802_964_live',
}


def resolve_device_specs(cmd_args) -> List[str]:
	if getattr(cmd_args, "cuda_devices", None):
		candidates = [token.strip() for token in cmd_args.cuda_devices.split(",") if token.strip()]
		if not candidates:
			raise ValueError("--cuda-devices provided but no valid GPU ids parsed.")
		return candidates
	num_gpus = getattr(cmd_args, "num_gpus", None)
	if num_gpus is None:
		return []
	if num_gpus < 0:
		raise ValueError("--num-gpus must be non-negative")
	if num_gpus == 0:
		return [""]
	return [str(idx) for idx in range(num_gpus)]


def chunk_dialogs(dialogs: Sequence[Tuple[str, dict]], num_chunks: int) -> List[List[Tuple[str, dict]]]:
	if num_chunks <= 1:
		return [list(dialogs)]
	chunked: List[List[Tuple[str, dict]]] = []
	chunk_size = math.ceil(len(dialogs) / num_chunks)
	for idx in range(num_chunks):
		start = idx * chunk_size
		end = min(len(dialogs), (idx + 1) * chunk_size)
		chunked.append(list(dialogs[start:end]))
	return chunked


def build_components(cmd_args, exp_1: DialogSession, sys_da, user_da):
	if cmd_args.llm in ['code-davinci-002']:
		backbone_model = OpenAIModel(cmd_args.llm)
		SysModel = PersuaderModel
		UsrModel = PersuadeeModel
		SysPlanner = P4GSystemPlanner
	elif cmd_args.llm in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'gpt2':
		model_source = cmd_args.local_model_path or 'gpt2'
		backbone_model = LocalModel(model_source, trust_remote_code=cmd_args.local_trust_remote_code)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'qwen2.5-0.5b-instruct':
		backbone_model = LocalModel('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'local':
		if not cmd_args.local_model_path:
			raise ValueError("--local-model-path is required when --llm local")
		backbone_model = LocalModel(cmd_args.local_model_path, trust_remote_code=cmd_args.local_trust_remote_code)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	else:
		raise ValueError(f"Unsupported llm option: {cmd_args.llm}")

	system = SysModel(
		sys_da,
		backbone_model,
		conv_examples=[exp_1],
		inference_args={
			"max_new_tokens": 80,
			"temperature": 0.7,
			"do_sample": True,
			"return_full_text": False,
		}
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
		conv_examples=[exp_1]
	)
	planner = SysPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_1]
	)
	game = PersuasionGame(system, user)
	return game, system, user, planner, backbone_model


def evaluate_dialogs(
	cmd_args,
	dialog_items: Sequence[Tuple[str, dict]],
	*,
	device_spec: Optional[str] = None,
	show_progress: bool = True,
) -> List[dict]:
	if device_spec is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = device_spec
		logger.info("Process %s restricting CUDA to: %s", os.getpid(), device_spec if device_spec else "CPU")

	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR

	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)
	game, system, user, planner, backbone_model = build_components(cmd_args, exp_1, sys_da, user_da)

	args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
	})

	results: List[dict] = []
	iterator = tqdm(dialog_items, desc="evaluating", disable=not show_progress)
	for did, dialog in iterator:
		if did in BAD_DIALOG_IDS:
			logger.debug("Skipping dialog %s due to safety filter", did)
			continue

		context = ""
		state = game.init_dialog()
		for t, turn in enumerate(dialog["dialog"]):
			if len(turn["ee"]) == 0:
				break
			if t == len(dialog["dialog"]) - 1:
				break

			usr_utt = " ".join(turn["ee"]).strip()
			usr_da = dialog["label"][t]["ee"][-1]

			if usr_da == "disagree-donation":
				usr_da = PersuasionGame.U_NoDonation
			elif usr_da == "negative-reaction-to-donation":
				usr_da = PersuasionGame.U_NegativeReaction
			elif usr_da == "positive-reaction-to-donation":
				usr_da = PersuasionGame.U_PositiveReaction
			elif usr_da == "agree-donation":
				usr_da = PersuasionGame.U_Donate
			else:
				usr_da = PersuasionGame.U_Neutral

			if usr_da == PersuasionGame.U_Donate:
				break

			sys_utt = " ".join(turn["er"]).strip()
			sys_da = set(dialog["label"][t]["er"])
			intersected_das = sys_da.intersection(system.dialog_acts)
			if len(intersected_das) == 0:
				sys_da_mapped = "other"
			else:
				sys_da_mapped = list(intersected_das)[-1]

			state.add_single(PersuasionGame.SYS, sys_da_mapped, sys_utt)
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)

			context = f"""
			{context}
			Persuader: {sys_utt}
			Persuadee: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			if isinstance(backbone_model, OpenAIModel):
				backbone_model._cached_generate.cache_clear()
			dialog_planner = OpenLoopMCTS(game, planner, args)
			for _ in range(args.num_MCTS_sims):
				dialog_planner.search(state)

			mcts_policy = dialog_planner.get_action_prob(state)
			mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]
			mcts_pred_rep = dialog_planner.get_best_realization(state, np.argmax(mcts_policy))

			human_resp = " ".join(dialog["dialog"][t + 1]["er"]).strip()
			next_sys_das = set(dialog["label"][t + 1]["er"])
			next_intersected_das = next_sys_das.intersection(system.dialog_acts)
			if len(next_intersected_das) == 0:
				next_sys_da = "other"
			else:
				next_sys_da = list(next_intersected_das)[-1]

			debug_data = {
				"probs": mcts_policy,
				"da": mcts_policy_next_da,
				"search_tree": {
					"Ns": dialog_planner.Ns,
					"Nsa": dialog_planner.Nsa,
					"Q": dialog_planner.Q,
					"P": dialog_planner.P,
					"Vs": dialog_planner.Vs,
					"realizations": dialog_planner.realizations,
					"realizations_Vs": dialog_planner.realizations_Vs,
					"realizations_Ns": dialog_planner.realizations_Ns,
				},
			}

			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': mcts_pred_rep,
				'new_da': mcts_policy_next_da,
				"debug": debug_data,
			}
			results.append(cmp_data)

			if cmd_args.debug:
				logger.debug(context)
				logger.debug("human resp: %s", human_resp)
				logger.debug("human da: %s", next_sys_da)
				logger.debug("mcts resp: %s", mcts_pred_rep)
				logger.debug("mcts da: %s", mcts_policy_next_da)

	return results


def _worker_entry(worker_id: int, device_spec: Optional[str], dialogs: Sequence[Tuple[str, dict]], cmd_args, result_queue: mp.Queue) -> None:
	try:
		res = evaluate_dialogs(cmd_args, dialogs, device_spec=device_spec, show_progress=False)
		result_queue.put((worker_id, res, None))
	except Exception:  # pragma: no cover - propagate worker failure
		import traceback
		result_queue.put((worker_id, None, traceback.format_exc()))


def main(cmd_args):
	with open("data/p4g/300_dialog_turn_based.pkl", "rb") as f:
		all_dialogs = pickle.load(f)

	ordered_dialogs: List[Tuple[str, dict]] = []
	for did, dialog in all_dialogs.items():
		if did in BAD_DIALOG_IDS:
			continue
		ordered_dialogs.append((did, dialog))
		if cmd_args.num_dialogs > 0 and len(ordered_dialogs) >= cmd_args.num_dialogs:
			break

	if not ordered_dialogs:
		logger.warning("No dialogs selected for evaluation.")
		return

	device_specs = resolve_device_specs(cmd_args)
	if not device_specs:
		device_specs = [None]

	if len(device_specs) == 1:
		results = evaluate_dialogs(
			cmd_args,
			ordered_dialogs,
			device_spec=device_specs[0],
			show_progress=True,
		)
	else:
		dialog_chunks = chunk_dialogs(ordered_dialogs, len(device_specs))
		queue: mp.Queue = mp.Queue()
		processes: List[Tuple[int, mp.Process]] = []
		active_workers: List[int] = []
		for worker_id, (device_spec, chunk) in enumerate(zip(device_specs, dialog_chunks)):
			if not chunk:
				continue
			proc = mp.Process(target=_worker_entry, args=(worker_id, device_spec, chunk, cmd_args, queue))
			proc.start()
			processes.append((worker_id, proc))
			active_workers.append(worker_id)

		collected: dict[int, List[dict]] = {}
		for _ in active_workers:
			worker_id, data, error = queue.get()
			if error is not None:
				for _, proc in processes:
					proc.terminate()
				for _, proc in processes:
					proc.join()
				raise RuntimeError(f"Worker {worker_id} failed with error:\n{error}")
			collected[worker_id] = data or []

		for _, proc in processes:
			proc.join()

		results: List[dict] = []
		for idx, chunk in enumerate(dialog_chunks):
			if not chunk:
				continue
			results.extend(collected.get(idx, []))

	with open(cmd_args.output, "wb") as f:
		pickle.dump(results, f)
	logger.info("Saved %d dialog evaluations to %s", len(results), cmd_args.output)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=str, default="", help='output file (autofills if empty)')
	parser.add_argument('--llm', type=str, default="code-davinci-002", choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo", "gpt2", "qwen2.5-0.5b-instruct", "local"], help='Backbone model identifier')
	parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
	parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
	parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
	parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
	parser.add_argument('--num_dialogs', type=int, default=100, help='number of dialogs to test MCTS on')
	parser.add_argument('--debug', action='store_true', help='debug mode')
	parser.add_argument('--local-model-path', type=str, default='', help='Path to a local Hugging Face model to load when using --llm gpt2 or --llm local.')
	parser.add_argument('--local-trust-remote-code', action='store_true', help='Allow executing remote code when loading local Hugging Face model.')
	parser.add_argument('--num-gpus', type=int, default=None, help='Limit number of CUDA devices to use (spawns one worker per GPU).')
	parser.add_argument('--cuda-devices', type=str, default=None, help='Comma-separated CUDA device ids to assign to workers. Overrides --num-gpus.')
	cmd_args = parser.parse_args()

	if not cmd_args.output:
		llm_label = cmd_args.llm
		model_label = Path(cmd_args.local_model_path).name if cmd_args.local_model_path else "base"
		cmd_args.output = (
			f"outputs/gdpzero-{cmd_args.num_mcts_sims}sims-"
			f"{llm_label}-{model_label}-{cmd_args.Q_0:.2f}Q-{cmd_args.num_dialogs}.pkl"
		)

	print("saving to", cmd_args.output)

	main(cmd_args)
