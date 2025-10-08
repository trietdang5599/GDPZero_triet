import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from trl import DPOTrainer

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_data(item):
    return {
        'prompt': 'Instruct: ' + item['prompt'] + '\n',
        'chosen': 'Output: ' + item['chosen'],
        'rejected': 'Output: ' + item['rejected']
    }

def train(model, ref_model, dataset, tokenizer, beta, training_args):
    model.train()
    ref_model.eval()

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_length=1024,
        max_prompt_length=512
    )

    dpo_trainer.train()

def build_preference_dataset(args):
    dialog_path = Path(args.dialog_path)
    if not dialog_path.exists():
        raise FileNotFoundError(f"Dialog dataset not found at {dialog_path}")

    with dialog_path.open("rb") as f:
        dialogs = pickle.load(f)

    rng = random.Random(args.seed)

    all_system_responses = []
    for sample in dialogs.values():
        for turn in sample.get("dialog", []):
            sys_utts = " ".join(turn.get("er") or []).strip()
            if sys_utts:
                all_system_responses.append(sys_utts)

    if not all_system_responses:
        raise ValueError("No system utterances found in the dialog dataset.")

    header = (
        "You are the Persuader. Continue the conversation in a way that persuades the "
        "Persuadee to donate to Save the Children.\nConversation so far:\n"
    )

    output_records = []
    for dialog_id, sample in dialogs.items():
        history_lines: List[str] = []
        for turn in sample.get("dialog", []):
            sys_utts = " ".join(turn.get("er") or []).strip()
            user_utts = " ".join(turn.get("ee") or []).strip()

            if sys_utts:
                prompt_context = "\n".join(history_lines).strip()
                prompt = header
                if prompt_context:
                    prompt += prompt_context + "\n"
                prompt += "Persuader:"

                for _ in range(args.num_negatives):
                    rejected = rng.choice(all_system_responses)
                    retry = 0
                    while rejected == sys_utts and retry < 5:
                        rejected = rng.choice(all_system_responses)
                        retry += 1
                    output_records.append(
                        {
                            "dialog_id": dialog_id,
                            "prompt": prompt,
                            "chosen": sys_utts,
                            "rejected": rejected,
                        }
                    )
                history_lines.append(f"Persuader: {sys_utts}")

            if user_utts:
                history_lines.append(f"Persuadee: {user_utts}")

    if not output_records:
        raise ValueError("No preference records were generated.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in output_records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(output_records)} preference pairs to {output_path}")


def parse_training_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb_project", type=str, default="truthy-dpo")
    return parser.parse_args(argv)


def run_training(args):
    seed_everything(args.seed)

    wandb.login()
    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(preprocess_data)

    training_args = TrainingArguments(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        report_to="wandb",
        output_dir='./results',
        logging_steps=10,
        remove_unused_columns=False,
    )

    train(model, ref_model, dataset, tokenizer, args.beta, training_args)

    model.save_pretrained("model-HF-DPO.pt")


def parse_build_args(argv):
    parser = argparse.ArgumentParser(
        description="Build preference dataset from P4G dialog pickle."
    )
    parser.add_argument("--dialog-path", required=True, help="Path to dialog pickle file.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--num-negatives", type=int, default=1, help="Negatives per prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args(argv)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "build-preference-dataset":
        args = parse_build_args(argv[1:])
        build_preference_dataset(args)
        return

    args = parse_training_args(argv)
    run_training(args)

if __name__ == "__main__":
    main()
