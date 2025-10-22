"""
Prompt builders for persona generation tasks.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Dict, Iterable, List, Optional


PERSONA_TYPE_SPECS: List[Dict[str, object]] = [
	{
		"field": "big_five_personality",
		"label": "Big-Five Personality",
		"options": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
	},
	{
		"field": "decision_making_style",
		"label": "Decision-Making Styles",
		"options": ["directive", "analytical", "conceptual", "behavioral"],
	},
]


PERSONA_FIELD_ORDER: List[str] = [spec["field"] for spec in PERSONA_TYPE_SPECS]


def _format_options(options: Iterable[str]) -> str:
	return '["' + '", "'.join(options) + '"]'


def format_persona_catalog() -> str:
	lines = ["********", "Persona types"]
	for spec in PERSONA_TYPE_SPECS:
		label = spec["label"]  # type: ignore[index]
		options = spec["options"]  # type: ignore[index]
		lines.append(f'{label}: {_format_options(options)}')
	lines.append("********")
	return "\n".join(lines)


SINGLE_PERSONA_PROMPT_TEMPLATE = dedent(
	"""
	You need to incorporate the following persona attributes and generate a cohesive persona description.
	Select exactly one attribute label from each persona type and keep the description easy to understand.
	Ensure the persona's age is between 28 and 40, and diversify occupations between personas.
	{persona_catalog}

	Please create fictional user profile {persona_id}{total_note}.
	{remaining_note}{history_block}
	Output requirements:
	- Return a JSON object (no surrounding array or code fences).
	- Include the keys:
	  * "id" (use exactly {persona_id})
	  * "big_five_personality" (string; choose exactly one from ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"])
	  * "decision_making_style" (string; choose exactly one from ["directive", "analytical", "conceptual", "behavioral"])
	  * "description" (a single paragraph similar to: "You are a 28-year-old female software developer...". Mention the selected attributes explicitly, keep the age between 28 and 40, and vary the occupation across personas.)
	- Attribute values must match one of the listed identifiers verbatim, not boolean maps or lists.
	"""
).strip()


def build_single_persona_prompt(
	persona_id: str,
	used_personas: Optional[List[Dict[str, str]]] = None,
	remaining_count: Optional[int] = None,
	total_count: Optional[int] = None,
	max_history: int = 5,
) -> str:
	persona_catalog = format_persona_catalog()
	if total_count is not None and total_count > 0:
		total_note = f" (profile {persona_id.split('_')[-1]} of {total_count})"
	else:
		total_note = ""

	remaining_note = ""
	if remaining_count is not None and remaining_count > 0:
		remaining_note = f"Personas left to produce after this: {max(remaining_count - 1, 0)}.\n"

	history_block = ""

	return SINGLE_PERSONA_PROMPT_TEMPLATE.format(
		persona_catalog=persona_catalog,
		persona_id=persona_id,
		total_note=total_note,
		remaining_note=remaining_note,
		history_block=history_block,
	)
