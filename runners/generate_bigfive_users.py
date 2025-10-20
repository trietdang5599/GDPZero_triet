#!/usr/bin/env python3

"""
Generate a set of user personas with Big Five traits by prompting an LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from core.gen_models import LocalModel
from core.model_factory import create_factor_llm
from utils.big_five_prompt import PERSONA_TYPE_SPECS, build_single_persona_prompt


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	"""Build the argparse schema and return parsed CLI arguments."""
	parser = argparse.ArgumentParser(
		description="Generate Big Five user personas via language model prompting."
	)
	parser.add_argument(
		"--llm",
		type=str,
		default="gpt2",
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
		"--local-base-model",
		type=str,
		default="",
		help="Base model path required when loading a PEFT adapter via --local-model-path.",
	)
	parser.add_argument(
		"--local-trust-remote-code",
		action="store_true",
		help="Allow execution of remote code when loading local HF models.",
	)
	parser.add_argument(
		"--num-personas",
		type=int,
		default=20,
		help="Number of user personas to request from the model.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.7,
		help="Sampling temperature passed to the generation model.",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=512,
		help="Maximum number of new tokens to sample from the model.",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=2,
		help="Additional attempts with increased token budget when the response is truncated.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("outputs/bigfive_personas.jsonl"),
		help="Destination .jsonl file for the generated personas.",
	)
	parser.add_argument(
		"--prompt-output",
		type=Path,
		default=None,
		help="Optional path to save the rendered prompt (defaults to <output>.prompt.txt).",
	)
	parser.add_argument(
		"--raw-output",
		type=Path,
		default=None,
		help="Optional path to save the raw model response (defaults to <output>.response.txt).",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Allow overwriting the output file if it already exists.",
	)
	parser.add_argument(
		"--show-prompt",
		action="store_true",
		help="Print the rendered prompt to stdout before querying the model.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Render the prompt and exit without contacting any model.",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
		help="Logging verbosity.",
	)
	return parser.parse_args()


def _ensure_parent_dir(path: Path) -> None:
	"""Create parent directories for the provided file path."""
	path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_output_path(base_path: Path, suffix: str) -> Path:
	"""Return the path with the requested suffix, adding an extension if missing."""
	if base_path.suffix:
		return base_path.with_suffix(suffix)
	return base_path.parent / f"{base_path.name}{suffix}"


def _call_generation_model(
	model,
	prompt: str,
	max_new_tokens: int,
	temperature: float,
	chat_messages: list[Dict[str, str]] | None = None,
) -> tuple[str, list[Dict[str, str]] | None]:
	"""Invoke the text or chat generation model and return raw text plus updated chat history."""
	inference_args: Dict[str, Any] = {
		"max_new_tokens": max_new_tokens,
		"temperature": temperature,
		"do_sample": temperature > 0,
		"return_full_text": False,
	}
	if isinstance(model, LocalModel):
		pad_token_id = getattr(getattr(model, "tokenizer", None), "pad_token_id", None)
		eos_token_id = getattr(getattr(model, "tokenizer", None), "eos_token_id", None)
		if pad_token_id is not None:
			inference_args["pad_token_id"] = pad_token_id
		if eos_token_id is not None:
			inference_args["eos_token_id"] = eos_token_id
	if hasattr(model, "chat_generate"):
		if chat_messages is None:
			messages = [
				{
					"role": "system",
					"content": "You are an expert annotator building diverse personas for persuasion research.",
				},
				{"role": "user", "content": prompt},
			]
		else:
			messages = chat_messages
		response = model.chat_generate(messages, **inference_args)
		text = response[0].get("generated_text", "").strip()
		messages.append({"role": "assistant", "content": text})
		return text, messages
	else:
		response = model.generate(prompt, **inference_args)
	if hasattr(model, "_cleaned_resp"):
		cleaned = model._cleaned_resp(response, prompt)  # type: ignore[attr-defined]
		if cleaned:
			return cleaned[0], None
	return response[0].get("generated_text", "").strip(), None

def _extract_json_object(payload: str) -> Dict[str, Any]:
	"""Extract a JSON object from model output, tolerating code fences and trailing commas."""
	text = payload.strip()
	if not text:
		raise ValueError("Model returned an empty response.")
	if text.startswith("```"):
		lines = [
			line
			for line in text.splitlines()
			if not line.strip().startswith("```")
		]
		text = "\n".join(lines).strip()
	first_brace = text.find("{")
	last_brace = text.rfind("}")
	if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
		snippet = text[first_brace : last_brace + 1]
		try:
			return json.loads(snippet)
		except json.JSONDecodeError:
			snippet = re.sub(r",(\s*[}\]])", r"\1", snippet)
			try:
				return json.loads(snippet)
			except json.JSONDecodeError:
				pass
	if text.startswith("{") and not text.strip().endswith("}"):
		raise ValueError(
			"Model response appears truncated (missing closing brace). "
			"Try increasing --max-new-tokens or rerunning generation."
		)
	try:
		return json.loads(text)
	except json.JSONDecodeError as exc:
		cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
		if cleaned != text:
			try:
				return json.loads(cleaned)
			except json.JSONDecodeError:
				pass
		raise ValueError(f"Failed to decode model output as JSON: {exc}") from exc


def _normalize_trait_value(value: Any, allowed_values: Sequence[str]) -> str:
	"""Coerce dict/list/str responses into a single allowed attribute string."""
	if isinstance(value, str):
		cleaned = value.strip()
		if cleaned in allowed_values:
			return cleaned
		raise ValueError(f"Value '{cleaned}' not in {list(allowed_values)}.")
	if isinstance(value, dict):
		candidates = [str(k).strip() for k, v in value.items() if v]
		if len(candidates) == 1 and candidates[0] in allowed_values:
			return candidates[0]
	if isinstance(value, (list, tuple, set)):
		candidates = [str(item).strip() for item in value if str(item).strip()]
		if len(candidates) == 1 and candidates[0] in allowed_values:
			return candidates[0]
	raise ValueError(f"Expected a single string value from {list(allowed_values)}, got {value!r}.")


def _accepted_attributes() -> Dict[str, set[str]]:
	"""Return the set of allowed attribute identifiers for each trait group."""
	return {
		spec["label"]: set(spec["options"])  # type: ignore[index]
		for spec in PERSONA_TYPE_SPECS
	}


def _validate_persona(
	persona: Dict[str, Any],
	allowed: Dict[str, set[str]],
	trait_map: Dict[str, str],
) -> None:
	"""Validate required fields and ensure every trait value is in the allowed set."""
	required_atomic_keys = ["id", "description"]
	missing_atomic = [key for key in required_atomic_keys if key not in persona]
	if missing_atomic:
		raise ValueError(f"Persona is missing required fields: {missing_atomic}")
	for alias, canonical in trait_map.items():
		if alias not in persona and canonical in persona:
			persona[alias] = persona[canonical]
		if alias not in persona:
			raise ValueError(f"Persona is missing required trait: {alias}")

	for alias, canonical in trait_map.items():
		value = str(persona[alias]).strip()
		if value not in allowed[canonical]:
			raise ValueError(
				f"Invalid attribute '{value}' for trait '{canonical}'. "
				f"Accepted: {sorted(allowed[canonical])}"
			)
	if not str(persona["description"]).strip():
		raise ValueError("Persona 'description' must be a non-empty string.")


def _canonical_trait_map() -> Dict[str, str]:
	"""Provide a mapping from JSON keys to the human-readable trait names."""
	return {
		spec["field"]: spec["label"]  # type: ignore[index]
		for spec in PERSONA_TYPE_SPECS
	}


def _validate_personas(personas: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Run validation on each persona and add metadata used downstream."""
	persona_list = list(personas)
	allowed = _accepted_attributes()
	trait_map = _canonical_trait_map()
	validated: List[Dict[str, Any]] = []
	for idx, persona in enumerate(persona_list, start=1):
		_validate_persona(persona, allowed, trait_map)
		for alias, canonical in trait_map.items():
			persona[canonical] = persona[alias]
		if "index" not in persona:
			persona["index"] = idx
		validated.append(persona)
	return validated


def _write_personas(personas: List[Dict[str, Any]], destination: Path) -> None:
	"""Write personas to disk in JSON Lines format at the given destination."""
	_ensure_parent_dir(destination)
	with destination.open("w", encoding="utf-8") as handle:
		for persona in personas:
			json.dump(persona, handle, ensure_ascii=False)
			handle.write("\n")


def _generate_persona(
	backbone_model,
	prompt: str,
	persona_id: str,
	allowed: Dict[str, set[str]],
	trait_map: Dict[str, str],
	max_new_tokens: int,
	max_retries: int,
	temperature: float,
	raw_output_path: Path,
) -> Dict[str, Any]:
	"""Generate a single persona, retrying until a valid JSON object is produced."""
	max_tokens = max(64, max_new_tokens)
	retry_count = 0
	retry_reason: str | None = None
	chat_messages: List[Dict[str, str]] | None = None
	combined_payload = ""
	call_index = 0

	while True:
		if chat_messages is not None and retry_reason:
			if retry_reason == "truncated":
				chat_messages.append(
					{
						"role": "user",
						"content": (
							"Continue the JSON object exactly where it stopped. "
							"Do not repeat earlier text or add commentary."
						),
					}
				)
			elif retry_reason == "invalid_attr":
				valid_traits = "; ".join(
					f"{canonical}: {sorted(values)}"
					for canonical, values in allowed.items()
				)
				chat_messages.append(
					{
						"role": "user",
						"content": (
							"One or more trait attributes were not chosen from the approved list or were formatted incorrectly. "
							"Respond again using only the permitted attribute identifiers as plain strings (not objects, arrays, or booleans). "
							f"Valid options per trait are: {valid_traits}. "
							"Return a single corrected JSON object."
						),
					}
				)
			retry_reason = None

		call_index += 1
		raw_chunk, updated_messages = _call_generation_model(
			backbone_model,
			prompt,
			max_new_tokens=max_tokens,
			temperature=temperature,
			chat_messages=chat_messages,
		)
		if updated_messages is not None:
			chat_messages = updated_messages
		if combined_payload and raw_chunk.lstrip().startswith("{"):
			combined_payload = raw_chunk
		else:
			combined_payload += raw_chunk
		with raw_output_path.open("a", encoding="utf-8") as raw_file:
			raw_file.write(
				f"--- persona {persona_id} call {call_index} ---\n{raw_chunk.strip()}\n\n"
			)

		try:
			persona = _extract_json_object(combined_payload)
		except ValueError as exc:
			if "truncated" in str(exc).lower():
				if retry_count >= max(0, max_retries):
					raise
				retry_count += 1
				retry_reason = "truncated"
				combined_payload = ""
				if isinstance(backbone_model, LocalModel):
					max_tokens *= 2
				else:
					max_tokens = int(max_tokens * 1.5)
				continue
			raise

		persona["id"] = persona_id
		try:
			persona["big_five_personality"] = _normalize_trait_value(
				persona.get("big_five_personality"),
				sorted(allowed["Big-Five Personality"]),
			)
			persona["decision_making_style"] = _normalize_trait_value(
				persona.get("decision_making_style"),
				sorted(allowed["Decision-Making Styles"]),
			)
		except (KeyError, ValueError) as exc:
			if retry_count >= max(0, max_retries):
				raise
			retry_count += 1
			retry_reason = "invalid_attr"
			combined_payload = ""
			continue
		try:
			_validate_persona(persona, allowed, trait_map)
		except ValueError as exc:
			if "Invalid attribute" in str(exc):
				if retry_count >= max(0, max_retries):
					raise
				retry_count += 1
				retry_reason = "invalid_attr"
				combined_payload = ""
				continue
			raise
		for alias, canonical in trait_map.items():
			persona[canonical] = persona[alias]
		return persona


def main() -> None:
	"""Command-line entry point orchestrating persona generation and persistence."""
	args = parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

	if args.output.exists() and not args.overwrite:
		raise FileExistsError(f"{args.output} already exists. Use --overwrite to replace it.")

	prompt_record_path = args.prompt_output or _normalize_output_path(args.output, ".prompt.txt")
	raw_output_path = args.raw_output or _normalize_output_path(args.output, ".response.txt")
	for aux_path in (prompt_record_path, raw_output_path):
		_ensure_parent_dir(aux_path)
	if args.dry_run:
		sample_prompt = build_single_persona_prompt(
			"user_001",
			[],
			remaining_count=args.num_personas,
			total_count=args.num_personas,
		)
		print("-------- Example persona prompt --------")
		print(sample_prompt)
		print("----------------------------------------")
		return

	prompt_record_path.write_text("", encoding="utf-8")
	raw_output_path.write_text("", encoding="utf-8")

	backbone_model, *_ = create_factor_llm(args)
	logger.info("Using generation model: %s", args.llm)

	allowed = _accepted_attributes()
	trait_map = _canonical_trait_map()
	used_personas: List[Dict[str, Any]] = []
	collected_personas: List[Dict[str, Any]] = []

	for idx in range(args.num_personas):
		persona_id = f"user_{idx + 1:03d}"
		remaining = args.num_personas - idx
		prompt = build_single_persona_prompt(
			persona_id,
			used_personas,
			remaining_count=remaining,
			total_count=args.num_personas,
		)
		if args.show_prompt and idx == 0:
			print("-------- Persona prompt --------")
			print(prompt)
			print("--------------------------------")
		with prompt_record_path.open("a", encoding="utf-8") as prompt_file:
			prompt_file.write(f"--- persona {persona_id} prompt ---\n{prompt}\n\n")

		persona = _generate_persona(
			backbone_model=backbone_model,
			prompt=prompt,
			persona_id=persona_id,
			allowed=allowed,
			trait_map=trait_map,
			max_new_tokens=args.max_new_tokens,
			max_retries=args.max_retries,
			temperature=args.temperature,
			raw_output_path=raw_output_path,
		)
		persona["index"] = idx + 1
		collected_personas.append(persona)
		used_personas.append(persona)

	personas = _validate_personas(collected_personas)
	_write_personas(personas, args.output)
	logger.info("Wrote %s personas to %s", len(personas), args.output)


if __name__ == "__main__":
	main()
