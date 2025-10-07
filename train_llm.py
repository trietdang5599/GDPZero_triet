#!/usr/bin/env python3
"""Utility to fine-tune causal LLMs on PersuasionForGood-style dialogues."""

import argparse
import json
import pickle
import random
import inspect
import warnings
from collections import defaultdict
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import os
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import IntervalStrategy
from datasets import Dataset as HFDataset

try:
    from trl import DPOTrainer, DPOConfig
except ImportError:  # pragma: no cover - optional dependency
    DPOTrainer = None

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

@dataclass
class ConversationExample:
    prompt: str
    completion: str
    dialog_id: str
    turn_index: int


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    dialog_id: str
    turn_index: int


def load_raw_records(dataset_path: Path) -> List[Dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".pkl":
        with dataset_path.open("rb") as f:
            payload = pickle.load(f)
    elif suffix in {".json", ".js"}:
        with dataset_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    elif suffix in {".jsonl", ".ndjson"}:
        records: List[Any] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        payload = records
    else:
        raise ValueError(f"Unsupported dataset extension: {dataset_path.suffix}")

    if isinstance(payload, dict):
        records = []
        for key, value in payload.items():
            record = {"id": key}
            if isinstance(value, dict):
                record.update(value)
            else:
                record["dialog"] = value
            records.append(record)
        return records
    if isinstance(payload, list):
        normalised = []
        for idx, value in enumerate(payload):
            if isinstance(value, dict):
                rec = {**value}
            else:
                rec = {"dialog": value}
            rec.setdefault("id", str(idx))
            normalised.append(rec)
        return normalised
    raise ValueError("Dataset must be a mapping or a list of dialog records.")


def join_utterances(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, Sequence):
        parts = [str(x).strip() for x in raw if str(x).strip()]
        return " ".join(parts)
    return str(raw).strip()


def build_examples(
    records: Iterable[Dict[str, Any]],
    *,
    system_field: str,
    user_field: str,
    system_role: str,
    user_role: str,
) -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    for record in records:
        dialog = record.get("dialog")
        if not isinstance(dialog, list):
            continue
        history_lines: List[str] = []
        for turn_idx, turn in enumerate(dialog):
            if not isinstance(turn, dict):
                continue
            sys_text = join_utterances(turn.get(system_field))
            usr_text = join_utterances(turn.get(user_field))
            if sys_text:
                history_lines.append(f"{system_role}: {sys_text}")
            if usr_text:
                history_lines.append(f"{user_role}: {usr_text}")
            next_turn = dialog[turn_idx + 1] if turn_idx + 1 < len(dialog) else None
            if not usr_text or not isinstance(next_turn, dict):
                continue
            next_sys = join_utterances(next_turn.get(system_field))
            if not next_sys:
                continue
            context_text = "\n".join(history_lines)
            prompt = f"{context_text}\n{system_role}: " if context_text else f"{system_role}: "
            examples.append(
                ConversationExample(
                    prompt=prompt,
                    completion=next_sys,
                    dialog_id=str(record.get("id", turn_idx)),
                    turn_index=turn_idx + 1,
                )
            )
    return examples

# issue might be here
def build_preference_examples(
    records: Iterable[Dict[str, Any]],
    *,
    system_field: str,
    user_field: str,
    system_role: str,
    user_role: str,
    rng: random.Random,
) -> List[PreferenceExample]:
    records = list(records)
    # direct_pairs = [rec for rec in records if rec.get("ori_resp") and rec.get("new_resp")]

    # if direct_pairs:
    #     preference_examples: List[PreferenceExample] = []
    #     for idx, rec in enumerate(direct_pairs):
    #         context = str(rec.get("context", "")).strip()
    #         chosen = str(rec.get("ori_resp", "")).strip()
    #         rejected = str(rec.get("new_resp", "")).strip()
    #         if not chosen or not rejected or chosen == rejected:
    #             continue
    #         prompt = context
    #         if prompt:
    #             prompt = prompt.rstrip()
    #             if not prompt.endswith("\n"):
    #                 prompt += "\n"
    #             if not prompt.strip().endswith(f"{system_role}:"):
    #                 prompt = f"{prompt}{system_role}: "
    #         else:
    #             prompt = f"{system_role}: "
    #         preference_examples.append(
    #             PreferenceExample(
    #                 prompt=prompt,
    #                 chosen=f" {chosen}",
    #                 rejected=f" {rejected}",
    #                 dialog_id=str(rec.get("did", idx)),
    #                 turn_index=int(rec.get("turn_index", idx)),
    #             )
    #         )
    #     if not preference_examples:
    #         raise ValueError("No valid preference pairs with ori/new responses found in dataset.")
    #     return preference_examples

    conversation_examples = build_examples(
        records,
        system_field=system_field,
        user_field=user_field,
        system_role=system_role,
        user_role=user_role,
    )
    if len(conversation_examples) < 2:
        raise ValueError("Need at least two conversation examples to build preference pairs for DPO.")

    pool = [ex.completion for ex in conversation_examples]
    if len(set(pool)) < 2:
        raise ValueError("Preference pool contains only one unique completion; cannot form rejected samples.")

    by_dialog: Dict[str, List[ConversationExample]] = {}
    for example in conversation_examples:
        by_dialog.setdefault(example.dialog_id, []).append(example)

    used_rejection_counts: Dict[str, int] = defaultdict(int)

    def normalize_text(text: str) -> str:
        return " ".join(text.split()).lower()

    def pick_rejected(example: ConversationExample) -> str:
        # Prefer negatives from the same dialogue to keep context mismatch realistic.
        in_dialog_candidates = [
            candidate.completion
            for candidate in by_dialog.get(example.dialog_id, [])
            if candidate.turn_index != example.turn_index
        ]
        cross_dialog_candidates = [
            candidate.completion
            for candidate in conversation_examples
            if candidate.dialog_id != example.dialog_id
        ]
        candidate_sets = [in_dialog_candidates, cross_dialog_candidates]
        chosen_norm = normalize_text(example.completion)
        for candidates in candidate_sets:
            if not candidates:
                continue
            unique_ordered = list(dict.fromkeys(candidates))
            local_valid: List[Tuple[int, str, str]] = []
            for rejected in unique_ordered:
                rejected_norm = normalize_text(rejected)
                if not rejected_norm or rejected_norm == chosen_norm:
                    continue
                similarity = SequenceMatcher(None, example.completion, rejected).ratio()
                if similarity >= 0.95:
                    continue
                usage = used_rejection_counts[rejected_norm]
                local_valid.append((usage, rejected, rejected_norm))
            if local_valid:
                min_usage = min(item[0] for item in local_valid)
                reservoir = [item for item in local_valid if item[0] == min_usage]
                _, selected_text, selected_norm = rng.choice(reservoir)
                used_rejection_counts[selected_norm] += 1
                return selected_text
        return ""

    preference_examples: List[PreferenceExample] = []
    for example in conversation_examples:
        rejected = pick_rejected(example)
        if not rejected:
            continue
        preference_examples.append(
            PreferenceExample(
                prompt=example.prompt,
                chosen=example.completion,
                rejected=rejected,
                dialog_id=example.dialog_id,
                turn_index=example.turn_index,
            )
        )
    if not preference_examples:
        raise ValueError("Failed to generate any preference examples; try adjusting dataset or sampling strategy.")
    return preference_examples


class ConversationDataset(Dataset):
    def __init__(
        self,
        data: Sequence[ConversationExample],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.data = list(data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def _encode_pair(self, prompt: str, completion: str) -> Dict[str, torch.Tensor]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            completion_ids = [self.tokenizer.eos_token_id]
        if completion_ids[-1] != self.tokenizer.eos_token_id:
            completion_ids = completion_ids + [self.tokenizer.eos_token_id]
        max_prompt_len = max(self.max_length - len(completion_ids), 0)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
        combined = prompt_ids + completion_ids
        combined = combined[-self.max_length :]
        labels = [-100] * len(prompt_ids) + completion_ids
        if len(labels) > self.max_length:
            labels = labels[-self.max_length :]
        attention_mask = [1] * len(combined)
        pad_length = self.max_length - len(combined)
        if pad_length > 0:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            combined = combined + [pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            labels = labels + [-100] * pad_length
        return {
            "input_ids": torch.tensor(combined, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.data[index]
        return self._encode_pair(example.prompt, example.completion)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LLM on dialogue data (SFT or DPO).")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/p4g/300_dialog_turn_based.pkl"), help="Path to dialogue dataset (.pkl, .json, .jsonl).")
    parser.add_argument("--model-name", type=str, default="gpt2", help="Hugging Face model identifier to fine-tune.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save checkpoints (defaults to outputs/<model>-<algorithm>).")
    parser.add_argument("--algorithm", type=str, choices=["sft", "dpo"], default="sft", help="Training objective.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for training tokens.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Linear warmup ratio.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Fraction of samples reserved for validation.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on total samples (0 = use all).")
    parser.add_argument("--system-field", type=str, default="er", help="Field name for system utterances in the dataset.")
    parser.add_argument("--user-field", type=str, default="ee", help="Field name for user utterances in the dataset.")
    parser.add_argument("--system-role", type=str, default="Persuader", help="Role label for system turns.")
    parser.add_argument("--user-role", type=str, default="Persuadee", help="Role label for user turns.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision if available.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Trainer logging frequency.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Max number of saved checkpoints.")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="Inverse temperature for the DPO loss.")
    parser.add_argument("--reference-model-name", type=str, default=None, help="Optional reference model identifier for DPO.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.output_dir is None:
        auto_dir = f"outputs/{args.model_name.replace('/', '_')}-{args.algorithm}"
        args.output_dir = Path(auto_dir)

    records = load_raw_records(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is not None and model.config.pad_token_id != tokenizer.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    max_positions = getattr(model.config, "max_position_embeddings", args.max_length)
    effective_max_length = min(args.max_length, max_positions)
    if effective_max_length < args.max_length:
        warnings.warn(
            f"Requested max_length={args.max_length} exceeds model capacity ({max_positions}). "
            f"Using {effective_max_length} instead.",
            RuntimeWarning,
        )
    else:
        effective_max_length = args.max_length

    if args.algorithm == "sft":
        examples = build_examples(
            records,
            system_field=args.system_field,
            user_field=args.user_field,
            system_role=args.system_role,
            user_role=args.user_role,
        )
        if not examples:
            raise ValueError("No training examples constructed from dataset.")

        random.shuffle(examples)
        if args.max_samples and args.max_samples > 0:
            examples = examples[: args.max_samples]

        val_size = int(len(examples) * args.validation_ratio)
        val_examples = examples[:val_size] if val_size > 0 else []
        train_examples = examples[val_size:]

        train_dataset = ConversationDataset(train_examples, tokenizer, effective_max_length)
        eval_dataset = ConversationDataset(val_examples, tokenizer, effective_max_length) if val_examples else None

        training_args = TrainingArguments(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            # dataloader_num_workers=2,
            # eval_strategy="epoch" if eval_dataset is not None else "no",
            # save_strategy="epoch",
            # save_total_limit=args.save_total_limit,
            # report_to="none",
            # fp16=args.fp16 and torch.cuda.is_available(),
            # 🔧 tránh treo do DataLoader
            dataloader_num_workers=0,           # debug/stable nhất
            dataloader_drop_last=True,          # batch lẻ → bỏ (tránh process nào đó thiếu batch)
            dataloader_pin_memory=False,        # giảm treo do pinned mem

            # ✅ tên tham số đúng
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            report_to="none",

            # DDP flags
            ddp_backend="nccl",
            ddp_find_unused_parameters=False,

            # Precision (bật fp16 nếu cậu đã confirm OK)
            fp16=(args.fp16 and torch.cuda.is_available()),
            bf16=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )

        unwrap_sig = inspect.signature(trainer.accelerator.unwrap_model)
        if "keep_torch_compile" not in unwrap_sig.parameters:
            original_unwrap = trainer.accelerator.unwrap_model

            def unwrap_model_compat(model, *args, **kwargs):
                kwargs.pop("keep_torch_compile", None)
                return original_unwrap(model, *args, **kwargs)

            trainer.accelerator.unwrap_model = unwrap_model_compat

        trainer.train()
        trainer.save_model()
    else:
        if DPOTrainer is None:
            raise ImportError("trl is required for DPO training. Install it with `pip install trl`. ")

        rng = random.Random(args.seed)
        pref_examples = build_preference_examples(
            records,
            system_field=args.system_field,
            user_field=args.user_field,
            system_role=args.system_role,
            user_role=args.user_role,
            rng=rng,
        )
        if not pref_examples:
            raise ValueError("No preference examples constructed from dataset.")
        for ex in pref_examples[:3]:
            print(f"Prompt: {ex.prompt}\nChosen: {ex.chosen}\nRejected: {ex.rejected}\n---")
            
        rng.shuffle(pref_examples)
        if args.max_samples and args.max_samples > 0:
            pref_examples = pref_examples[: args.max_samples]

        val_size = int(len(pref_examples) * args.validation_ratio)
        val_pref = pref_examples[:val_size] if val_size > 0 else []
        train_pref = pref_examples[val_size:]
        if not train_pref:
            raise ValueError("Not enough preference examples for training after splitting.")

        train_dataset = HFDataset.from_list(
            [{"prompt": ex.prompt, "chosen": ex.chosen, "rejected": ex.rejected} for ex in train_pref]
        )
        eval_dataset = (
            HFDataset.from_list(
                [{"prompt": ex.prompt, "chosen": ex.chosen, "rejected": ex.rejected} for ex in val_pref]
            )
            if val_pref
            else None
        )

        reference_name = args.reference_model_name or args.model_name
        reference_model = AutoModelForCausalLM.from_pretrained(reference_name)
        if tokenizer.pad_token_id is not None and reference_model.config.pad_token_id != tokenizer.pad_token_id:
            reference_model.resize_token_embeddings(len(tokenizer))
            reference_model.config.pad_token_id = tokenizer.pad_token_id

        do_eval = eval_dataset is not None
        dpo_args = DPOConfig(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            eval_strategy=IntervalStrategy.EPOCH if eval_dataset is not None else IntervalStrategy.NO,
            save_strategy=IntervalStrategy.EPOCH,
            save_total_limit=args.save_total_limit,
            report_to=[],
            fp16=args.fp16 and torch.cuda.is_available(),
            bf16=False,
            remove_unused_columns=False,
            beta=args.dpo_beta,
            max_length=effective_max_length,
            max_prompt_length=min(effective_max_length, args.max_length),
            gradient_checkpointing=False,
            do_train=True,
            do_eval=do_eval,
        )

        dpo_trainer = DPOTrainer(
            model,
            reference_model,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        dpo_trainer.train()
        dpo_trainer.save_model()

    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
