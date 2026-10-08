"""
Self-contained script to demonstrate inference using the best trained JEV checkpoint.
Loads Qwen3-0.6B backbone, applies LoRA adapters, loads the trained choice-scoring head,
and answers sample evaluation questions (choice, yes/no, score).
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure bev-train is in Python path for utility imports if present
workspace_dir = Path(__file__).resolve().parent.parent
bev_train_dir = workspace_dir / "bev-train"
if bev_train_dir.exists() and str(bev_train_dir) not in sys.path:
    sys.path.insert(0, str(bev_train_dir))

from dataset import TASK_TYPES, question_to_choices, collate_fn
from tokenization import encode_example
from logger import load_checkpoint

def run_sample_inference():
    checkpoint_path = Path(__file__).parent / "checkpoints" / "best"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Using device: {device}")

    network, meta = load_checkpoint(str(checkpoint_path), device)
    network.eval()
    print("Checkpoint and backbone loaded successfully!\n")

    # Sample decision context
    state = (
        "Customer message: 'I ordered the silver wireless headphones 5 days ago, "
        "and the tracking status has not updated since day 1. I need them for my flight tomorrow. "
        "Can I get an immediate replacement or full refund?'"
    )

    sample_questions = [
        {
            "type": "noul",
            "instructions": "Is the customer asking for an immediate resolution?",
            "choices": ["No", "Yes"]
        },
        {
            "type": "choice",
            "instructions": "What is the best customer support triage category?",
            "criteria": {
                "shipping_delay": "Inquire with logistics and offer expedited replacement",
                "technical_support": "Troubleshoot hardware issue",
                "billing_error": "Fix invoice or incorrect charge"
            }
        },
        {
            "type": "score",
            "instructions": "Rate customer urgency on a scale from 0 (low) to 4 (extreme).",
            "choices": ["0 - Low", "1 - Moderate", "2 - High", "3 - Very High", "4 - Extreme"]
        }
    ]

    for q in sample_questions:
        print(f"--- Question Type: {q['type']} ---")
        print(f"Instruction: {q['instructions']}")
        choices, _ = question_to_choices(q)
        example = encode_example(
            state,
            q["instructions"],
            choices,
            q["type"],
            meta["max_state_tokens"],
            meta["max_choice_tokens"]
        )
        example.update(task_type=TASK_TYPES[q["type"]], label=None)
        batch = {k: v.to(device) for k, v in collate_fn([example]).items()}

        with torch.no_grad():
            logits = network(**batch)
            probs = torch.softmax(logits[0].float(), dim=-1).tolist()

        if q["type"] == "noul":
            p_yes = probs[1]
            print(f"Answer: {'YES' if p_yes > 0.5 else 'NO'} (Confidence: {p_yes*100:.1f}% Yes, {probs[0]*100:.1f}% No)\n")
        elif q["type"] == "choice":
            keys = list(q["criteria"].keys())
            best_idx = max(range(len(keys)), key=lambda i: probs[i])
            print(f"Selected Choice: {keys[best_idx]} (prob={probs[best_idx]:.3f})")
            for k, p in zip(keys, probs):
                print(f"  - {k}: {p*100:.1f}%")
            print()
        elif q["type"] == "score":
            expected_score = sum(i * p for i, p in enumerate(probs))
            print(f"Predicted Score: {expected_score:.2f} / 4.0")
            for i, p in enumerate(probs):
                print(f"  - Score {i}: {p*100:.1f}%")
            print()

if __name__ == "__main__":
    run_sample_inference()
