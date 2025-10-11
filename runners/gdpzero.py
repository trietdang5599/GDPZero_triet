import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pickle
import argparse
from datetime import datetime
import numpy as np

from tqdm.auto import tqdm
from core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.P4GSystemPlanner import P4GSystemPlanner, P4GChatSystemPlanner
from core.PersuaderModel import PersuaderModel, PersuaderChatModel
from core.PersuadeeModel import PersuadeeModel, PersuadeeChatModel
from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR
	
	logger.info("System dialog acts: %s", sys_da)
	logger.info("User dialog acts: %s", user_da)
	

	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)
	

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
	elif cmd_args.llm in ['qwen2.5-0.5b','qwen2.5-7b', 'llamda-3-8b', 'deepseek-r1']:
		if cmd_args.llm == 'qwen2.5-7b':
			model_source = 'Qwen/Qwen2.5-7B-Instruct'
		elif cmd_args.llm == 'qwen2.5-0.5b':
			model_source = 'Qwen/Qwen2.5-0.5B-Instruct'
		elif cmd_args.llm == 'llamda-3-8b':
			model_source = 'meta-llama/Meta-Llama-3-8B-Instruct'
		elif cmd_args.llm == 'deepseek-r1':
			model_source = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
		backbone_model = LocalModel(model_source, trust_remote_code=True)
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
		raise ValueError(f"Unsupported llm: {cmd_args.llm}")
	
	system = SysModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1],
		inference_args={
			"max_new_tokens": 80,
			"temperature": 0.7,
			"do_sample": True,  # for MCTS open loop
			"return_full_text": False,
		}
	)
	user = UsrModel(
		user_da,
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,  # for MCTS open loop
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

	logger.debug("System dialog acts (model): %s", system.dialog_acts)
	logger.debug("User dialog acts (model): %s", user.dialog_acts)

	with open("data/p4g/300_dialog_turn_based.pkl", "rb") as f:
		all_dialogs = pickle.load(f)

	num_dialogs = cmd_args.num_dialogs
	args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
	})

	output = []  # for evaluation. [{did, context, ori_resp, new_resp, debug}, ...]
	# those dialogs has inappropriated content and will throw an error/be filtered with OPENAI models. See raw_prompting.py file for more details
	bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']  # throws exception due to ChatGPT API filtering
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating")
	for did in all_dialogs.keys():
		if did in bad_dialogs:
			logger.debug("Skipping dialog id: %s", did)
			continue
		if num_done == num_dialogs:
			break

		logger.info("Evaluating dialog id: %s", did)
		context = ""
		dialog = all_dialogs[did]
		
		state = game.init_dialog()
		donation_success = False
		for t, turn in enumerate(dialog["dialog"]):
			if len(turn["ee"]) == 0:  # ended
				break
			# also skip last turn as there is no evaluation
			if t == len(dialog["dialog"]) - 1:
				break

			usr_utt = " ".join(turn["ee"]).strip()
			usr_da = dialog["label"][t]["ee"][-1]

			# map to our dialog act
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

			# map sys as well
			sys_utt = " ".join(turn["er"]).strip()
			sys_da = set(dialog["label"][t]["er"])
			intersected_das = sys_da.intersection(system.dialog_acts)
			if len(intersected_das) == 0:
				sys_da = "other"
			else:
				sys_da = list(intersected_das)[-1]
			
			state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)

			# update context for evaluation
			context = f"""
			{context}
			Persuader: {sys_utt}
			Persuadee: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			if usr_da == PersuasionGame.U_Donate:
				logger.info(
					"Dialog %s success: persuadee agreed to donate at turn %s with response: %s",
					did,
					t,
					usr_utt,
				)
				donation_success = True
				break

			# mcts policy
			if isinstance(backbone_model, OpenAIModel):
				backbone_model._cached_generate.cache_clear()
			dialog_planner = OpenLoopMCTS(game, planner, args)
			logger.debug("Searching action for turn %s", t)
			for i in tqdm(range(args.num_MCTS_sims)):
				dialog_planner.search(state)

			mcts_policy = dialog_planner.get_action_prob(state, did)
			mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]

			print("mcts_policy_next_da: ", mcts_policy_next_da)

			# # fetch the generated utterance from simulation
			mcts_pred_rep = dialog_planner.get_best_realization(state, np.argmax(mcts_policy))

			# next ground truth utterance
			human_resp = " ".join(dialog["dialog"][t+1]["er"]).strip()
			next_sys_das = set(dialog["label"][t+1]["er"])
			next_intersected_das = next_sys_das.intersection(system.dialog_acts)
			if len(next_intersected_das) == 0:
				next_sys_da = "other"
			else:
				next_sys_da = list(next_intersected_das)[-1]

			# logging for debug
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

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': mcts_pred_rep,
				'new_da': mcts_policy_next_da,
				"debug": debug_data,
			}
			output.append(cmp_data)

			# if cmd_args.debug:
			logger.info("Context:\n%s", context)
			logger.info("human resp: %s", human_resp)
			logger.info("human da: %s", next_sys_da)
			logger.info("mcts resp: %s", mcts_pred_rep)
			logger.info("mcts da: %s", mcts_policy_next_da)
		# final of the game
		final_outcome = 1.0 if donation_success else game.get_dialog_ended(state)
		logger.info("Dialog %s final outcome: %s", did, final_outcome)
		with open(cmd_args.output, "wb") as f:
			pickle.dump(output, f)
		num_done += 1
		pbar.update(1)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=str, default="", help='output file (autofills if empty)')
	parser.add_argument('--llm', type=str, default="code-davinci-002", choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo", "gpt2", "qwen2.5-0.5b", "qwen2.5-7b", "llamda-3-8b", "deepseek-r1", "local"], help='Backbone model identifier')
	parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
	parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
	parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
	parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
	parser.add_argument('--num_dialogs', type=int, default=100, help='number of dialogs to test MCTS on')
	parser.add_argument('--debug', action='store_true', help='debug mode')
	parser.add_argument('--log-level', type=str, default='INFO', choices=['CRITICAL','ERROR','WARNING','INFO','DEBUG','NOTSET'], help='Logging level for terminal output.')
	parser.add_argument('--local-model-path', type=str, default='', help='Path to a local Hugging Face model to load when using --llm gpt2 or --llm local.')
	parser.add_argument('--local-trust-remote-code', action='store_true', help='Allow executing remote code when loading local Hugging Face model.')
	cmd_args = parser.parse_args()

	log_dir = Path("logs")
	log_dir.mkdir(parents=True, exist_ok=True)
	log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
	file_handler = logging.FileHandler(log_path, encoding="utf-8")
	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(getattr(logging, cmd_args.log_level.upper(), logging.INFO))
	logging.basicConfig(
		level=logging.DEBUG,
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
		handlers=[stream_handler, file_handler],
	)
	if cmd_args.debug:
		stream_handler.setLevel(logging.DEBUG)
	logger.info("Writing logs to %s", log_path)

	if not cmd_args.output:
		llm_label = cmd_args.llm
		model_label = Path(cmd_args.local_model_path).name if cmd_args.local_model_path else "base"
		cmd_args.output = (
			f"outputs/gdpzero-{cmd_args.num_mcts_sims}sims-"
			f"{llm_label}-{model_label}-{cmd_args.Q_0:.2f}Q-{cmd_args.num_dialogs}.pkl"
		)

	logger.info("Saving to %s", cmd_args.output)
	main(cmd_args)
