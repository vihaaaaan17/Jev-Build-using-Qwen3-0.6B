"""
Self-contained script to demonstrate inference using the trained JEV checkpoint.
Loads Qwen3-0.6B backbone, applies LoRA adapters, loads the trained choice-scoring head,
and answers sample evaluation questions (choice, yes/no, score).
Completely standalone with zero external local repository dependencies.
"""

import json
import math
from pathlib import Path
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3Model

TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.new_dim = new_dim
        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        self.o = nn.Linear(new_dim, hidden_dim)

    def forward(self, x, padding_mask):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.einsum("bqd,bkd->bqk", [q, k]) / math.sqrt(self.new_dim)
        scores = scores.masked_fill(~padding_mask[:, None, :], -float("inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bqk,bkd->bqd", [weights, v])
        return self.o(out)

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, new_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, new_dim),
            nn.GELU(),
            nn.Linear(new_dim, hidden_dim),
        )

    def forward(self, x, padding_mask):
        x = x + self.attn(self.norm1(x), padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class ChoiceHead(nn.Module):
    def __init__(self, hidden_dim, new_dim, num_layers, num_task_types):
        super().__init__()
        self.new_dim = new_dim
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, new_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, new_dim)
        self.choice_proj = nn.Linear(hidden_dim, new_dim)

    def forward(self, choice_embeddings, answer_embedding, choice_mask, task_type):
        B = choice_embeddings.shape[0]
        task_token = self.task_embedding(task_type).unsqueeze(1)
        x = torch.cat([task_token, choice_embeddings, answer_embedding.unsqueeze(1)], dim=1)
        x = self.input_norm(x)

        always_valid = torch.ones(B, 1, dtype=torch.bool, device=choice_mask.device)
        padding_mask = torch.cat([always_valid, choice_mask, always_valid], dim=1)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)

        answer = self.answer_proj(x[:, -1])
        choices = self.choice_proj(x[:, 1:-1])
        logits = torch.einsum("bd,bcd->bc", [answer, choices]) / math.sqrt(self.new_dim)
        return logits.masked_fill(~choice_mask, -float("inf"))

class BEVNetwork(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, position_ids, attention_mask, choice_read_idx, choice_mask, answer_end_idx, task_type, **kwargs):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False
        )
        hidden = output.last_hidden_state.float()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)

        choice_embeddings = hidden[batch_idx[:, None], choice_read_idx]
        answer_embedding = hidden[batch_idx, answer_end_idx]
        return self.head(choice_embeddings, answer_embedding, choice_mask, task_type)

def load_checkpoint(ckpt_dir, device):
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)

    overrides = {"num_hidden_layers": meta.get("num_layers", 20)}
    backbone = Qwen3Model.from_pretrained(meta["model_name"], dtype=torch.float32, **overrides)
    if (ckpt_dir / "lora").exists():
        backbone = PeftModel.from_pretrained(backbone, ckpt_dir / "lora")
    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **meta["head_config"])
    head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))
    return BEVNetwork(backbone, head).to(device).eval(), meta

def question_to_choices(question):
    t = question["type"]
    if t == "choice":
        return list(question["criteria"].values()), question.get("label")
    elif t == "score":
        return question["choices"], question.get("label")
    elif t == "noul":
        return ["No", "Yes"], question.get("label")
    raise ValueError(f"Unknown task type: {t}")

def encode_example(tokenizer, state, instructions, choices, task_type, max_state_tokens, max_choice_tokens):
    task_prompts = {
        "choice": "Choose the best option.",
        "score": "Rate it on the given scale.",
        "noul": "Answer yes or no.",
    }
    def trunc(txt, limit):
        toks = tokenizer.encode(txt)
        return txt if len(toks) <= limit else tokenizer.decode(toks[:limit])

    question_text = f"{task_prompts[task_type]}\nQuestion: {instructions}\n\n{trunc(state, max_state_tokens)}\n"
    choices = [trunc(c, max_choice_tokens) for c in choices]

    choice_begin = tokenizer.encode("<option>")
    choice_end = tokenizer.encode("</option>")
    q_tokens = tokenizer.encode(question_text)
    ans_tokens = tokenizer.encode("The answer is:")
    c_tokens = [tokenizer.encode(c) for c in choices]

    final_tokens = list(q_tokens)
    choices_end_idx = []
    for c in c_tokens:
        final_tokens.extend(choice_begin)
        final_tokens.extend(c)
        final_tokens.extend(choice_end)
        choices_end_idx.append(len(final_tokens) - 1)
    final_tokens.extend(ans_tokens)
    ans_end_idx = len(final_tokens) - 1

    # Position IDs
    pos_ids = list(range(len(q_tokens)))
    choice_start_pos = len(q_tokens)
    for c_end in choices_end_idx:
        cur_len = c_end - (pos_ids[-1] if len(pos_ids) > len(q_tokens) else len(q_tokens) - 1)
        # Each choice starts at choice_start_pos
        # Compute choice token length
        # Using exact position mapping
    # Simplified clean generation
    pos_ids = list(range(len(q_tokens)))
    cur_end = len(q_tokens) - 1
    for c_end in choices_end_idx:
        span_len = c_end - cur_end
        pos_ids.extend([choice_start_pos + i for i in range(span_len)])
        cur_end = c_end
    ans_start = max(pos_ids) + 1
    for i in range(ans_end_idx + 1 - len(pos_ids)):
        pos_ids.append(ans_start + i)

    # 2D Attention Mask
    L = ans_end_idx + 1
    mask = torch.zeros((L, L))
    q_len = len(q_tokens)
    mask[:q_len, :q_len] = torch.tril(torch.ones(q_len, q_len))
    start_c = q_len
    for end_c in choices_end_idx:
        mask[start_c:end_c + 1, start_c:end_c + 1] = torch.tril(torch.ones(end_c - start_c + 1, end_c - start_c + 1))
        start_c = end_c + 1
    mask[q_len:, :q_len] = 1
    ans_start_idx = choices_end_idx[-1] + 1
    mask[ans_start_idx:, :ans_start_idx] = 1
    mask[ans_start_idx:, ans_start_idx:] = torch.tril(torch.ones(L - ans_start_idx, L - ans_start_idx))

    opt_len = len(tokenizer.encode("</option>"))
    read_idx = [e - opt_len for e in choices_end_idx]

    return {
        "input_ids": torch.tensor([final_tokens]),
        "position_ids": torch.tensor([pos_ids]),
        "attention_mask": mask.unsqueeze(0),
        "choice_read_idx": torch.tensor([read_idx]),
        "choice_mask": torch.ones((1, len(choices)), dtype=torch.bool),
        "answer_end_idx": torch.tensor([ans_end_idx]),
        "task_type": torch.tensor([TASK_TYPES[task_type]]),
    }

def run_sample_inference():
    checkpoint_path = Path(__file__).parent / "checkpoints" / "best"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Using device: {device}")

    network, meta = load_checkpoint(str(checkpoint_path), device)
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name"])
    network.eval()
    print("Checkpoint loaded successfully!\n")

    state = (
        "Customer message: 'I ordered the wireless headphones 5 days ago, "
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
        batch = encode_example(
            tokenizer, state, q["instructions"], choices, q["type"],
            meta["max_state_tokens"], meta["max_choice_tokens"]
        )
        batch = {k: v.to(device) for k, v in batch.items()}

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
