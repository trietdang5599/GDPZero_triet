import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from core.helpers import DialogSession

def set_determinitic_seed(seed):
	if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	return


class dotdict(dict):
	def __getattr__(self, name):
		return self[name]


class hashabledict(dict):
	def __hash__(self):
		return hash(tuple(sorted(self.items())))


logger = logging.getLogger(__name__)


def get_preference_pair(
	probabilities: np.ndarray,
	state_rep: str,
	dialog_acts: Iterable[str],
	valid_moves: Iterable[int],
	realizations_vs: Optional[Dict[str, Dict[str, float]]],
) -> Optional[Tuple[int, Tuple[str, float], Tuple[str, float]]]:
	"""Return the preference pair (best/worst samples) for the next strategy."""
	if not realizations_vs:
		return None

	probabilities = np.asarray(probabilities)
	if probabilities.size == 0:
		return None

	valid_moves_list = [int(action_idx) for action_idx in valid_moves]
	if not valid_moves_list:
		return None

	best_prob = -float("inf")
	target_idx = None
	for action_idx in valid_moves_list:
		prob_val = float(probabilities[action_idx])
		if prob_val > best_prob:
			best_prob = prob_val
			target_idx = action_idx

	if target_idx is None:
		return None

	dialog_acts_list = list(dialog_acts)
	if 0 <= target_idx < len(dialog_acts_list):
		label = dialog_acts_list[target_idx]
	else:
		label = str(target_idx)

	prefetch_key = f"{state_rep}__{label}"
	realization_dict = realizations_vs.get(prefetch_key)
	if not realization_dict or len(realization_dict) < 2:
		return None

	best_pair = max(realization_dict.items(), key=lambda kv: kv[1])
	worst_pair = min(realization_dict.items(), key=lambda kv: kv[1])
	return target_idx, best_pair, worst_pair


def summarize_action_statistics(
	probabilities: np.ndarray,
	state_rep: str,
	dialog_acts: Iterable[str],
	valid_moves: Iterable[int],
	q_table: Dict[str, Dict[int, float]],
	nsa_table: Dict[str, Dict[int, int]],
	realizations_vs: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple[str, str]:
	"""Build human-readable summaries for the action distribution explored by MCTS.

	Args:
		probabilities: Array of action probabilities (aligned with dialog_acts).
		state_rep: String representation of the current state.
		dialog_acts: Ordered iterable mapping action index -> dialog act label.
		valid_moves: Iterable of action indices that were considered valid.
		q_table: Mapping state -> {action: Q-value}.
		nsa_table: Mapping state -> {action: visit count}.
		realizations_vs: Optional mapping for OpenLoopMCTS storing realizations and their values.

	Returns:
		Tuple of (header_line, detail_block) suitable for logging.
	"""
	probabilities = np.asarray(probabilities)
	dialog_acts = list(dialog_acts)
	valid_moves = list(valid_moves)
	best_idx = int(np.argmax(probabilities)) if probabilities.size else None

	def _label_for(action_idx: int) -> str:
		if 0 <= action_idx < len(dialog_acts):
			return dialog_acts[action_idx]
		return str(action_idx)

	best_label = _label_for(best_idx) if best_idx is not None else "None"
	best_prob = float(probabilities[best_idx]) if best_idx is not None else None
	state_qs = q_table.get(state_rep, {})
	state_nsas = nsa_table.get(state_rep, {})
	best_q = state_qs.get(best_idx) if best_idx is not None else None
	best_visits = state_nsas.get(best_idx) if best_idx is not None else None
	best_prob_str = f"{best_prob:.3f}" if best_prob is not None else "None"
	best_q_str = f"{best_q:.4f}" if best_q is not None else "None"
	best_visits_str = str(best_visits) if best_visits is not None else "None"

	header = (
		f"Visit distribution: {[float(f'{p:.3f}') for p in probabilities]}; "
		f"best action={best_idx} ({best_label}) (prob={best_prob_str}, Q={best_q_str}, visits={best_visits_str})"
	)

	strategy_lines = []
	for action_idx in valid_moves:
		label = _label_for(action_idx)
		prob_val = float(probabilities[action_idx])
		sentence = "N/A"
		realization_info = []
		if realizations_vs:
			prefetch_key = f"{state_rep}__{label}"
			realization_dict = realizations_vs.get(prefetch_key)
			if realization_dict:
				best_utt, _ = max(realization_dict.items(), key=lambda kv: kv[1])
				sentence = best_utt
				realization_info = [
					f"- {utt} (v={val:.3f})" for utt, val in realization_dict.items()
				]
		entry = f"[{label} | {sentence} | {prob_val:.3f}]"
		if realization_info:
			entry += "\n\t" + "\n\t".join(realization_info)
		strategy_lines.append(entry)

	detail_block = "\n".join(strategy_lines)
	return header, detail_block


def export_preference_pair(
    dialog_id: str,
	state: DialogSession,
	preference_pair: Optional[Tuple[int, Tuple[str, float], Tuple[str, float]]],
	system_role: str,
	output_path: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
	"""Append the preference pair to preference_pair.json if it is new."""
	if not preference_pair:
		return None

	_, best_pair, worst_pair = preference_pair
	if not best_pair or not worst_pair:
		return None

	conversation = state.to_string_rep(keep_sys_da=False, keep_user_da=False)
	prompt_header = (
		"You are the Persuader. Continue the conversation in a way that persuades "
		"the Persuadee to donate to Save the Children."
	)
	if conversation:
		prompt = f"{prompt_header}\nConversation so far:\n{conversation}\n{system_role}:"
	else:
		prompt = f"{prompt_header}\nConversation so far:\n{system_role}:"

	# dialog_id = hashlib.sha1(hashable_state.encode("utf-8")).hexdigest()
	preference_entry = {
		"dialog_id": dialog_id,
		"prompt": prompt,
		"chosen": best_pair[0],
		"rejected": worst_pair[0],
	}

	output_path = output_path or Path(__file__).resolve().parents[1] / "preference_pair.jsonl"
	existing_entries = []
	if output_path.exists():
		try:
			with output_path.open("r", encoding="utf-8") as pref_file:
				raw_content = pref_file.read().strip()
			if raw_content:
				if raw_content.startswith("["):
					loaded = json.loads(raw_content)
					if isinstance(loaded, list):
						existing_entries = loaded
				else:
					for line in raw_content.splitlines():
						line = line.strip()
						if not line:
							continue
						try:
							entry = json.loads(line)
							if isinstance(entry, dict):
								existing_entries.append(entry)
						except json.JSONDecodeError:
							logger.warning("Failed to parse preference entry line; skipping.")
		except json.JSONDecodeError:
			logger.warning("%s is not valid JSON; resetting file.", output_path.name)
		except Exception as exc:
			logger.warning("Failed to read %s: %s", output_path.name, exc)

	def _entry_key(entry: Dict[str, str]) -> tuple[str, str, str, str]:
		return (
			entry.get("dialog_id", ""),
			entry.get("prompt", ""),
			entry.get("chosen", ""),
			entry.get("rejected", ""),
		)

	existing_keys = {_entry_key(entry) for entry in existing_entries}
	if _entry_key(preference_entry) not in existing_keys:
		existing_entries.append(preference_entry)
		with output_path.open("w", encoding="utf-8") as pref_file:
			for entry in existing_entries:
				pref_file.write(json.dumps(entry, ensure_ascii=True))
				pref_file.write("\n")
	return preference_entry
