import json
import pathlib

cells = []

def md(s):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": s.strip("\n").splitlines(True)
    })

def code(s):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s.strip("\n").splitlines(True)
    })

# ==========================================
# 0. HEADER & THEORY
# ==========================================
md("""
# Building JEV (Judge / Evaluator / Verifier) — Kaggle Standalone Edition
### Train Choice-Order Invariant Decision Models on Qwen3-0.6B with `avbiswas/bev-decision`

This notebook is **100% self-contained**: all modules, tokenization routines, architecture classes, and training loops are embedded directly in executable code blocks (no GitHub cloning or external scripts needed).

---

### Core Innovations
1. **Choice-Order Invariance in a Single Forward Pass:**
   Standard causal LLMs read tokens sequentially, meaning Option B attends to Option A, creating strong position/recency bias. Here, every choice is assigned the **exact same starting position IDs** immediately following the prompt.
2. **2D Custom Block Attention Mask:**
   - **Question/State** attends causally to itself.
   - **Each Choice** attends to itself and the Question/State, but **never to other choices**.
   - **Answer Tokens** at the end attend to all choices and the question.
3. **ChoiceHead Scoring:**
   Choice embeddings and the answer embedding are passed into an attention scoring head, producing bounded probabilities for:
   - **`CHOICE`**: Categorical multiple-choice selection
   - **`NOUL`**: Binary Yes/No proposition ($P(\\text{Yes})$)
   - **`SCORE`**: Ordinal scale scoring ($0..N$)
4. **Kaggle Free-Tier Hardware Optimization:**
   NVIDIA T4 / P100 GPUs do not feature native hardware `bfloat16`. This notebook includes dynamic hardware detection that automatically configures `float16` + `torch.amp.GradScaler` on T4/P100 (or `bfloat16` on Ampere/A100) to ensure zero gradient underflow and zero loss divergence.
5. **Interactive Progress & Hugging Face Hub Publishing:**
   Includes `tqdm` progress bars for batch tracking, metric streaming, and a dedicated code block to push the fine-tuned model and ChoiceHead directly to Hugging Face.

---

### Estimated Runtimes on Kaggle Free GPU (Tesla T4)
| Configuration | Training Samples | Optimizer Steps | Est. Runtime |
| :--- | :--- | :--- | :--- |
| **Sanity Smoke Test** | 10 questions (15 epochs) | 30 steps | **~20 – 30 seconds** |
| **Quick Experiment** | 5,000 questions (1 epoch) | 78 steps | **~4 – 5 minutes** |
| **Kaggle Production (Default)** | 20,000 questions (1 epoch) | 312 steps | **~18 – 22 minutes** |
| **Full Dataset** | 150,000 questions (1 epoch) | 2,343 steps | **~2.2 – 2.5 hours** |

*Note: A 20-minute run consumes less than 1% of your 30-hour weekly Kaggle GPU quota.*

---

### Kaggle Session Settings Before Running
1. In the right panel, open **Session options**.
2. **Accelerator**: Select **GPU T4 x2** (or GPU P100).
3. **Internet**: Toggle **On** (required to download weights and the dataset from Hugging Face).
""")

# ==========================================
# 1. DEPENDENCIES
# ==========================================
md("""
## 1. Install Dependencies
Installs modern, compatible versions of Hugging Face `transformers`, `datasets`, `peft`, `huggingface_hub`, and PyTorch acceleration libraries.
""")

code("""
# 1. Fix Kaggle torchao version mismatch:
# Kaggle environment comes with an outdated torchao 0.10.0 that causes PEFT to crash.
# We uninstall torchao (not needed for LoRA) and install/update required packages.
!pip uninstall -y torchao
!pip install -q "transformers>=4.48.0" "datasets>=2.20.0" "peft>=0.10.0" "huggingface_hub>=0.28.0" pyyaml accelerate safetensors tqdm
print("All dependencies installed successfully.")
""")

# ==========================================
# 2. HARDWARE & PRECISION SETUP
# ==========================================
md("""
## 2. Hardware Verification & Precision Configuration
Checks GPU capability and automatically configures either **FP16 with `GradScaler`** (for T4/P100, compute capability < 8) or **BF16 native autocast** (for A100/L4, compute capability >= 8).
""")

code("""
import os
import sys
import contextlib
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Active GPU: {device_name} (Compute Capability: {capability[0]}.{capability[1]})")
    if capability[0] < 8:
        print(">> GPU Capability < 8 detected (T4/P100): Will use FP16 + PyTorch GradScaler.")
    else:
        print(">> GPU Capability >= 8 detected: Will use BF16 native autocast.")
else:
    print("WARNING: No GPU detected! Please enable GPU accelerator in Kaggle session options.")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_precision_config(device):
    \"\"\"
    Determines precision mode:
    - CUDA Capability < 8 (T4, P100): float16, use_scaler=True
    - CUDA Capability >= 8 (A100, L4): bfloat16, use_scaler=False
    - Other devices: float32, use_scaler=False
    \"\"\"
    if device.type == "cuda":
        cap_major = torch.cuda.get_device_capability(device)[0]
        if cap_major < 8:
            return torch.float16, True
        return torch.bfloat16, False
    elif device.type == "mps":
        return torch.bfloat16, False
    return torch.float32, False

def autocast_ctx(device):
    dtype, _ = get_precision_config(device)
    if device.type in ("cuda", "mps") and dtype != torch.float32:
        return torch.autocast(device.type, dtype=dtype)
    return contextlib.nullcontext()

def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
""")

# ==========================================
# 3. TOKENIZATION & MASKING
# ==========================================
md("""
## 3. Tokenization, Position IDs & 2D Block Attention Mask
Implements the core order-invariant encoding mechanism:
* Choices are delimited by `<option>` and `</option>` tags.
* Position IDs for all choices start at the exact same index right after the question tokens.
* The attention mask enforces that choices cannot see each other, only the question/state.
""")

code("""
import os
import torch
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "Qwen/Qwen3-0.6B"
print(f"Loading tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

TASK_PROMPTS = {
    "choice": "Choose the best option.",
    "score": "Rate it on the given scale.",
    "noul": "Answer yes or no.",
}

def tokenize_prompt(question, choices):
    choice_begin_token = tokenizer.encode("<option>")
    choice_end_token = tokenizer.encode("</option>")
    answer_prompt = "The answer is:"

    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode(answer_prompt)
    choice_tokens = [tokenizer.encode(choice) for choice in choices]

    final_tokens = []
    final_tokens.extend(question_tokens)

    span = {
        "question_end_idx": len(question_tokens) - 1,
        "choices_end_idx": [],
        "answer_end_idx": 0,
    }

    for choice in choice_tokens:
        final_tokens.extend(choice_begin_token)
        final_tokens.extend(choice)
        final_tokens.extend(choice_end_token)
        span["choices_end_idx"].append(len(final_tokens) - 1)

    final_tokens.extend(answer_tokens)
    span["answer_end_idx"] = len(final_tokens) - 1

    return final_tokens, span

def generate_position_ids(span):
    \"\"\"
    All choices share the exact same starting position IDs immediately after the question.
    \"\"\"
    position_ids = []
    start_idx = 0
    for i in range(span["question_end_idx"] + 1):
        position_ids.append(start_idx)
        start_idx += 1

    next_choice_begin_idx = start_idx
    for choice_end_idx in span["choices_end_idx"]:
        for idx in range(choice_end_idx - next_choice_begin_idx + 1):
            position_ids.append(start_idx + idx)
        next_choice_begin_idx = choice_end_idx + 1

    answer_start_idx = max(position_ids) + 1
    for idx in range(span["answer_end_idx"] + 1 - len(position_ids)):
        position_ids.append(answer_start_idx + idx)

    return position_ids

def get_attention_mask(span):
    \"\"\"
    Constructs 2D attention mask:
    - Question attends causally to question.
    - Each choice attends causally to self and fully to question.
    - Choice i CANNOT attend to Choice j.
    - Answer tokens attend to everything.
    \"\"\"
    L = span["answer_end_idx"] + 1
    attention_mask = torch.zeros((L, L))
    question_len = span["question_end_idx"] + 1

    attention_mask[:question_len, :question_len] = torch.tril(torch.ones(question_len, question_len))

    choice_spans = []
    choice_start_idx = question_len
    for choice_end_idx in span["choices_end_idx"]:
        choice_spans.append((choice_start_idx, choice_end_idx))
        choice_start_idx = choice_end_idx + 1

    for choice_start_idx, choice_end_idx in choice_spans:
        span_len = choice_end_idx - choice_start_idx + 1
        attention_mask[choice_start_idx:choice_end_idx + 1, choice_start_idx:choice_end_idx + 1] = torch.tril(torch.ones(span_len, span_len))

    # Choices attend to question
    attention_mask[question_len:, :question_len] = 1

    # Answer tokens attend to all previous tokens and causally to themselves
    if choice_spans:
        answer_tokens_begin = choice_spans[-1][1] + 1
        attention_mask[answer_tokens_begin:, :answer_tokens_begin] = 1
        ans_len = L - answer_tokens_begin
        attention_mask[answer_tokens_begin:, answer_tokens_begin:] = torch.tril(torch.ones(ans_len, ans_len))

    return attention_mask

def truncate(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])

def build_question(state, instructions, task_type, max_state_tokens):
    return f"{TASK_PROMPTS[task_type]}\\nQuestion: {instructions}\\n\\n{truncate(state, max_state_tokens)}\\n"

def encode_example(state, instructions, choices, task_type, max_state_tokens, max_choice_tokens):
    question = build_question(state, instructions, task_type, max_state_tokens)
    choices = [truncate(choice, max_choice_tokens) for choice in choices]

    final_tokens, span = tokenize_prompt(question, choices)
    position_ids = generate_position_ids(span)
    attention_mask = get_attention_mask(span)

    choice_end_token_len = len(tokenizer.encode("</option>"))
    choice_read_idx = [end_idx - choice_end_token_len for end_idx in span["choices_end_idx"]]

    return {
        "input_ids": final_tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "choice_read_idx": choice_read_idx,
        "answer_end_idx": span["answer_end_idx"],
    }

print("Tokenization and masking functions defined.")
""")

# ==========================================
# 4. DATASET & COLLATOR
# ==========================================
md("""
## 4. Dataset Loading & Dynamic Collation
Loads [`avbiswas/bev-decision`](https://huggingface.co/datasets/avbiswas/bev-decision) from Hugging Face:
- Unpacks multi-question rows into individual sample questions with a progress bar.
- Formats labels for `choice`, `noul`, and `score` types.
- Dynamically collates batches with pad tokens and SDPA-ready 4D attention masks (`-inf` / `0.0`).
""")

code("""
import json
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

DATASET_NAME = "avbiswas/bev-decision"
FALLBACK_DATASET_NAME = "avbiswas/bev-decision-150K"

TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}

def question_to_choices(question):
    q_type = question["type"]
    if q_type == "choice":
        keys = list(question["criteria"])
        choices = [f"{key}: {question['criteria'][key]}" if question["criteria"][key] else key for key in keys]
        label = keys.index(question["label"]) if "label" in question else None
    elif q_type == "score":
        choices = [f"{i}: {level}" for i, level in enumerate(question["criteria"])]
        label = question.get("label")
    else:  # noul -> binary Yes/No
        choices = ["No", "Yes"]
        label = int(question["label"]) if "label" in question else None
    return choices, label

def explode_questions(rows):
    out = {"state": [], "task_type": [], "instructions": [], "choices": [], "label": []}
    for state, questions_json in zip(rows["state"], rows["questions_json"]):
        for question in json.loads(questions_json).values():
            choices, label = question_to_choices(question)
            out["state"].append(state)
            out["task_type"].append(TASK_TYPES[question["type"]])
            out["instructions"].append(question["instructions"])
            out["choices"].append(choices)
            out["label"].append(label)
    return out

def load_questions(split, max_questions=None, seed=0):
    print(f"Loading '{split}' split from {DATASET_NAME}...")
    try:
        ds = load_dataset(DATASET_NAME, split=split)
    except Exception as e:
        print(f"Loading {DATASET_NAME} failed ({e}), falling back to {FALLBACK_DATASET_NAME}...")
        ds = load_dataset(FALLBACK_DATASET_NAME, split=split)

    ds = ds.shuffle(seed=seed)
    if max_questions is not None:
        ds = ds.select(range(min(max_questions, len(ds))))
    ds = ds.map(explode_questions, batched=True, remove_columns=ds.column_names, desc=f"Exploding {split} questions")
    if max_questions is not None:
        ds = ds.select(range(min(max_questions, len(ds))))
    print(f"Loaded {len(ds):,} questions for split '{split}'.")
    return ds

class BEVDataset(torch.utils.data.Dataset):
    def __init__(self, questions, max_state_tokens, max_choice_tokens):
        self.questions = questions
        self.max_state_tokens = max_state_tokens
        self.max_choice_tokens = max_choice_tokens

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        row = self.questions[idx]
        example = encode_example(
            row["state"], row["instructions"], row["choices"], TASK_NAMES[row["task_type"]],
            self.max_state_tokens, self.max_choice_tokens,
        )
        example["task_type"] = row["task_type"]
        example["label"] = row["label"]
        return example

def collate_fn(batch):
    B = len(batch)
    L = max(len(ex["input_ids"]) for ex in batch)
    C = max(len(ex["choice_read_idx"]) for ex in batch)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    attention_mask = torch.eye(L).repeat(B, 1, 1)
    choice_read_idx = torch.zeros((B, C), dtype=torch.long)
    choice_mask = torch.zeros((B, C), dtype=torch.bool)
    answer_end_idx = torch.zeros(B, dtype=torch.long)

    for b, ex in enumerate(batch):
        n, c = len(ex["input_ids"]), len(ex["choice_read_idx"])
        input_ids[b, :n] = torch.tensor(ex["input_ids"])
        position_ids[b, :n] = torch.tensor(ex["position_ids"])
        attention_mask[b, :n, :n] = ex["attention_mask"]
        choice_read_idx[b, :c] = torch.tensor(ex["choice_read_idx"])
        choice_mask[b, :c] = True
        answer_end_idx[b] = ex["answer_end_idx"]

    # 4D attention mask for SDPA: (B, 1, L, L)
    attention_mask = torch.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1)

    out = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "choice_read_idx": choice_read_idx,
        "choice_mask": choice_mask,
        "answer_end_idx": answer_end_idx,
        "task_type": torch.tensor([ex["task_type"] for ex in batch], dtype=torch.long),
    }
    if any(ex["label"] is not None for ex in batch):
        out["labels"] = torch.tensor([ex["label"] if ex["label"] is not None else 0 for ex in batch], dtype=torch.long)
    return out

print("Dataset pipeline ready.")
""")

# ==========================================
# 5. MODEL ARCHITECTURE
# ==========================================
md("""
## 5. Model Architecture: Backbone + PEFT/LoRA + ChoiceHead
- **Backbone**: Qwen3-0.6B (loads only the first 20 decoder layers to save memory and compute).
- **LoRA**: Parameter-efficient fine-tuning on attention projections of the top kept layers.
- **ChoiceHead**: Runs self-attention over `[task_token, choice_embeddings..., answer_embedding]` and computes bilinear scores:
  $$\\text{logits}_c = \\frac{\\mathbf{v}_{\\text{ans}}^\\top \\mathbf{W}_{\\text{ans}}^\\top \\mathbf{W}_{\\text{choice}} \\mathbf{v}_{\\text{choice}_c}}{\\sqrt{D}}$$
  The ChoiceHead runs in FP32 for numerical stability.
""")

code("""
import math
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModel

# Kaggle environment compatibility patch: neutralize incompatible torchao in PEFT
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

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
            use_cache=False,
        )
        hidden = output.last_hidden_state.float()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)

        choice_embeddings = hidden[batch_idx[:, None], choice_read_idx]
        answer_embedding = hidden[batch_idx, answer_end_idx]

        # Head operates in fp32 for stable loss and gradients
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(choice_embeddings, answer_embedding, choice_mask, task_type)

def load_backbone(model_name, num_layers=None):
    overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
    try:
        from transformers import Qwen3Model
        return Qwen3Model.from_pretrained(model_name, dtype=torch.float32, **overrides)
    except Exception:
        return AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, **overrides)

def lora_target_regex(num_layers, last_k_layers):
    layer_ids = "|".join(str(i) for i in range(num_layers - last_k_layers, num_layers))
    return rf".*layers\\.({layer_ids})\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj)"

def build_network(model_name, num_layers, lora_r, lora_alpha, lora_dropout, lora_last_k_layers, head_config):
    backbone = load_backbone(model_name, num_layers)
    if lora_last_k_layers == 0:
        backbone.requires_grad_(False)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_regex(backbone.config.num_hidden_layers, lora_last_k_layers),
        )
        backbone = get_peft_model(backbone, lora_config)
    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **head_config)
    return BEVNetwork(backbone, head)

print("Model architecture ready.")
""")

# ==========================================
# 6. LOGGING & CHECKPOINTS
# ==========================================
md("""
## 6. Logging & Checkpoint Management
Saves training metrics to `runs/<exp_id>/metrics.jsonl` and serializes compact checkpoints (storing LoRA adapters + `head.pt` only, leaving base weights external).
""")

code("""
import time
from datetime import datetime
from pathlib import Path
import yaml
from safetensors.torch import load_file

class RunLogger:
    def __init__(self, config, run_name, device, root="runs"):
        self.config = config
        self.exp_id = f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}"
        self.run_dir = Path(root) / self.exp_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

        self.write_config(self.run_dir)
        with open(self.run_dir / "run.json", "w") as f:
            json.dump({"exp_id": self.exp_id, "device": str(device), "started_at": datetime.now().isoformat()}, f, indent=2)
        print(f"Logging initialized at: {self.run_dir}")

    def write_config(self, directory):
        with open(directory / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

    def log(self, step, **metrics):
        record = {"step": step, "time": round(time.time() - self.start_time, 1), **metrics}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\\n")
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    def save_checkpoint(self, network, name="final", step=None):
        ckpt_dir = self.run_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(network.backbone, PeftModel):
            network.backbone.save_pretrained(ckpt_dir / "lora")
        torch.save({k: v.cpu() for k, v in network.head.state_dict().items()}, ckpt_dir / "head.pt")

        self.write_config(ckpt_dir)
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump({
                "exp_id": self.exp_id,
                "step": step,
                "model_name": self.config["model"]["name"],
                "num_layers": network.backbone.config.num_hidden_layers,
                "head_config": {**self.config["head"], "num_task_types": network.head.task_embedding.num_embeddings},
                "max_state_tokens": self.config["data"]["max_state_tokens"],
                "max_choice_tokens": self.config["data"]["max_choice_tokens"],
            }, f, indent=2)
        print(f"Checkpoint saved -> {ckpt_dir}")
        return ckpt_dir

    def finish(self, **summary):
        summary["duration_sec"] = round(time.time() - self.start_time, 1)
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

def load_checkpoint(ckpt_dir, device):
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)

    backbone = load_backbone(meta["model_name"], meta.get("num_layers"))
    if (ckpt_dir / "lora").exists():
        backbone = PeftModel.from_pretrained(backbone, ckpt_dir / "lora")
    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **meta["head_config"])
    head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))
    return BEVNetwork(backbone, head).to(device).eval(), meta

print("Logger & Checkpoint utilities ready.")
""")

# ==========================================
# 7. EVALUATION & METRICS
# ==========================================
md("""
## 7. Evaluation & Metrics with Progress Bar
Computes cross-entropy loss, overall decision accuracy, task-stratified accuracy (`choice`, `score`, `noul`), and binary ROC-AUC for `noul` decisions with real-time `tqdm` tracking.
""")

code("""
from collections import defaultdict
import torch.nn.functional as F
from tqdm.auto import tqdm

def binary_auc(scores, labels):
    scores, labels = torch.tensor(scores), torch.tensor(labels, dtype=torch.bool)
    positives, negatives = scores[labels], scores[~labels].sort().values
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    below = torch.searchsorted(negatives, positives, right=False)
    below_or_tied = torch.searchsorted(negatives, positives, right=True)
    return (below + below_or_tied).sum().item() / 2 / (len(positives) * len(negatives))

@torch.no_grad()
def evaluate(network, dataloader, device, desc="Evaluating"):
    network.eval()
    total_loss, correct, count = 0.0, defaultdict(int), defaultdict(int)
    type_names = {v: k for k, v in TASK_TYPES.items()}
    noul_p_yes, noul_labels = [], []

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        batch = to_device(batch, device)
        with autocast_ctx(device):
            logits = network(**batch)
        total_loss += F.cross_entropy(logits, batch["labels"], reduction="sum").item()

        hits = logits.argmax(-1) == batch["labels"]
        for task_type, hit in zip(batch["task_type"].tolist(), hits.tolist()):
            count[type_names[task_type]] += 1
            correct[type_names[task_type]] += hit

        is_noul = batch["task_type"] == TASK_TYPES["noul"]
        if is_noul.any():
            noul_p_yes += torch.softmax(logits[is_noul].float(), dim=-1)[:, 1].tolist()
            noul_labels += batch["labels"][is_noul].tolist()

    n = max(sum(count.values()), 1)
    metrics = {"loss": total_loss / n, "accuracy": sum(correct.values()) / n}
    for name in count:
        metrics[f"accuracy_{name}"] = correct[name] / max(count[name], 1)
    if 0 < sum(noul_labels) < len(noul_labels):
        metrics["auc_noul"] = binary_auc(noul_p_yes, noul_labels)
    network.train()
    return metrics

def validate(network, test_loader, device, logger, step, best_accuracy):
    metrics = evaluate(network, test_loader, device, desc=f"Eval @ Step {step}")
    logger.log(step, split="test", **metrics)
    if metrics["accuracy"] > best_accuracy:
        best_accuracy = metrics["accuracy"]
        logger.save_checkpoint(network, name="best", step=step)
    return metrics, best_accuracy

print("Evaluation engine ready.")
""")

# ==========================================
# 8. TRAINING ENGINE
# ==========================================
md("""
## 8. Training Engine with Progress Bars & Adaptive Scaling
Integrates AdamW with split learning rates (LoRA vs ChoiceHead), cosine schedule with warmup, gradient clipping, gradient accumulation, adaptive `GradScaler` for T4/P100 GPUs, and live `tqdm` progress monitoring.
""")

code("""
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

def make_loader(split, config, device, shuffle):
    data = config["data"]
    max_questions = data["max_train_questions"] if split == "train" else data["max_test_questions"]
    dataset = BEVDataset(load_questions(split, max_questions, config["seed"]), data["max_state_tokens"], data["max_choice_tokens"])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=data.get("num_workers", 2),
        persistent_workers=(data.get("num_workers", 2) > 0),
        pin_memory=(device.type == "cuda"),
    )

def train(config, run_name="jev-run", resume=None):
    torch.manual_seed(config["seed"])
    device = get_device()
    _, use_scaler = get_precision_config(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if (use_scaler and device.type == "cuda") else None

    logger = RunLogger(config, run_name, device)
    model_cfg, lora_cfg, optim_cfg, log_cfg = config["model"], config["lora"], config["optim"], config["logging"]

    head_config = {**config["head"], "num_task_types": len(TASK_TYPES)}
    network = build_network(
        model_cfg["name"], model_cfg["num_layers"],
        lora_cfg["r"], lora_cfg["alpha"], lora_cfg["dropout"],
        lora_cfg["last_k_layers"], head_config
    )

    if resume:
        ckpt_dir = Path(resume)
        if (ckpt_dir / "lora").exists() and isinstance(network.backbone, PeftModel):
            weights = load_file(ckpt_dir / "lora" / "adapter_model.safetensors")
            network.backbone.load_state_dict(weights, strict=False)
        network.head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))
        print(f"Resumed weights from {ckpt_dir}")

    network = network.to(device)
    count = lambda params: sum(p.numel() for p in params if p.requires_grad)
    print(f"Trainable params: backbone={count(network.backbone.parameters()):,} | head={count(network.head.parameters()):,}")

    train_loader = make_loader("train", config, device, shuffle=True)
    test_loader = None
    if config["data"].get("max_test_questions", 0) != 0:
        test_loader = make_loader("test", config, device, shuffle=False)

    lora_params = [p for p in network.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": float(optim_cfg["lr_lora"])},
        {"params": network.head.parameters(), "lr": float(optim_cfg["lr_head"])},
    ], weight_decay=float(optim_cfg["weight_decay"]))

    accum = optim_cfg.get("grad_accum_steps", 1)
    effective_loader_len = (len(train_loader) // accum) * accum
    total_steps = optim_cfg["epochs"] * (len(train_loader) // accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(optim_cfg["warmup_ratio"] * total_steps), total_steps)

    step = 0
    best_accuracy = -1.0
    network.train()

    print(f"\\n=== Training Started ===")
    print(f"Epochs: {optim_cfg['epochs']} | Optimizer steps: {total_steps} | Micro-batches/opt-step: {accum}")

    for epoch in range(optim_cfg["epochs"]):
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader, 1), total=effective_loader_len, desc=f"Epoch {epoch+1}/{optim_cfg['epochs']}", leave=True)
        for micro, batch in pbar:
            if micro > effective_loader_len:
                break
            batch = to_device(batch, device)
            with autocast_ctx(device):
                logits = network(**batch)
            loss = F.cross_entropy(logits, batch["labels"])

            if scaler is not None:
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()

            correct = (logits.argmax(-1) == batch["labels"]).sum().item()
            epoch_loss += loss.item() * len(logits)
            epoch_correct += correct
            epoch_count += len(logits)

            # Update live tqdm postfix
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / len(logits):.3f}",
                "opt_step": step,
            })

            if micro % accum != 0:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), optim_cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(network.parameters(), optim_cfg["grad_clip"])
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log_cfg["log_every"] == 0:
                lrs = {"lr_head": scheduler.get_last_lr()[1]}
                if lora_params:
                    lrs["lr_lora"] = scheduler.get_last_lr()[0]
                logger.log(step, split="train", epoch=epoch, loss=loss.item(), accuracy=correct / len(logits), **lrs)

            if step % log_cfg["save_every"] == 0:
                logger.save_checkpoint(network, name="latest", step=step)

            if test_loader is not None and step % log_cfg["eval_every"] == 0:
                test_metrics, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)

        logger.log(step, split="train_epoch", epoch=epoch, loss=epoch_loss / max(epoch_count, 1),
                   accuracy=epoch_correct / max(epoch_count, 1))

    summary = {"final_train_loss": epoch_loss / max(epoch_count, 1), "final_train_accuracy": epoch_correct / max(epoch_count, 1)}
    if test_loader is not None:
        if step % log_cfg["eval_every"] != 0:
            test_metrics, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)
        summary["test"] = test_metrics
        summary["best_test_accuracy"] = best_accuracy

    ckpt_dir = logger.save_checkpoint(network, name="final", step=step)
    logger.finish(checkpoint=str(ckpt_dir), **summary)
    return network, ckpt_dir

print("Training engine ready.")
""")

# ==========================================
# 9. SANITY CHECK (SMOKE TEST)
# ==========================================
md("""
## 9. Quick Sanity Check (Smoke Test)
Overfits 10 questions to verify that loss decreases toward zero, accuracy rises to 1.0, and the entire pipeline functions end-to-end without errors (**~20 – 30 seconds**).
""")

code("""
smoke_config = {
    "seed": 0,
    "model": {"name": "Qwen/Qwen3-0.6B", "num_layers": 20},
    "data": {
        "max_train_questions": 10,
        "max_test_questions": 0,
        "max_state_tokens": 512,
        "max_choice_tokens": 64,
        "batch_size": 5,
        "num_workers": 0,
    },
    "lora": {"last_k_layers": 4, "r": 8, "alpha": 16, "dropout": 0.0},
    "head": {"new_dim": 512, "num_layers": 2},
    "optim": {
        "epochs": 15,
        "lr_lora": 1.0e-4,
        "lr_head": 1.0e-3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "grad_clip": 1.0,
        "grad_accum_steps": 1,
    },
    "logging": {"log_every": 2, "eval_every": 100, "save_every": 100},
}

print("Running Smoke Test (10 questions overfit check)...")
smoke_net, smoke_ckpt = train(smoke_config, run_name="smoke")
print("Smoke test completed successfully!")
""")

# ==========================================
# 10. KAGGLE PRODUCTION TRAINING
# ==========================================
md("""
## 10. Kaggle Free-Tier Full Training Run
Production configuration tailored for Kaggle's free GPU limits:
- **Questions**: 20,000 train questions, 500 test questions
- **Batching**: Micro-batch size 8, gradient accumulation 8 $\\rightarrow$ Effective batch size 64
- **Layers**: First 20 layers of Qwen3-0.6B with LoRA on layers 8–19
- **Runtime**: **~18 – 22 minutes** on a single Kaggle T4 GPU.
""")

code("""
kaggle_config = {
    "seed": 0,
    "model": {
        "name": "Qwen/Qwen3-0.6B",
        "num_layers": 20,
    },
    "data": {
        "max_train_questions": 50000,  # Scaled from 20K to 50K for higher coverage & accuracy
        "max_test_questions": 1000,    # 1,000 held-out evaluation questions
        "max_state_tokens": 768,       # Extended from 512 to 768 to avoid truncating reasoning context
        "max_choice_tokens": 64,
        "batch_size": 8,               # Safe for 16GB VRAM with FP16
        "num_workers": 2,
    },
    "lora": {
        "last_k_layers": 12,           # LoRA on layers 8-19
        "r": 16,                       # Doubled rank from 8 to 16 for higher capacity
        "alpha": 32,                   # Doubled alpha from 16 to 32
        "dropout": 0.05,               # Regularization to prevent overfitting over 2 epochs
    },
    "head": {
        "new_dim": 512,
        "num_layers": 2,
    },
    "optim": {
        "epochs": 1,                   # Keep 1 epoch: prevents overfitting, all 50k samples are fresh
        "lr_lora": 1.0e-4,
        "lr_head": 1.0e-3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "grad_clip": 1.0,
        "grad_accum_steps": 8,         # 8 x 8 = 64 questions per optimizer step (781 steps total)
    },
    "logging": {
        "log_every": 20,
        "eval_every": 250,
        "save_every": 250,
    },
}

print("Launching Kaggle Production Training...")
trained_network, final_ckpt = train(kaggle_config, run_name="kaggle-jev")
print(f"Training finished. Final checkpoint located at: {final_ckpt}")
""")

# ==========================================
# 11. INFERENCE & DECISION DEMO
# ==========================================
md("""
## 11. Live Interactive Inference API
The `answer()` function takes raw text input and question metadata, tokenizes with custom position IDs and attention masks, and returns typed, calibrated decision outputs across all 3 decision types (`CHOICE`, `NOUL`, `SCORE`).
""")

code("""
@torch.no_grad()
def answer(network, meta, state, question, device):
    network.eval()
    choices, _ = question_to_choices(question)
    example = encode_example(state, question["instructions"], choices, question["type"],
                             meta["max_state_tokens"], meta["max_choice_tokens"])
    example.update(task_type=TASK_TYPES[question["type"]], label=None)
    batch = to_device(collate_fn([example]), device)

    with autocast_ctx(device):
        probs = torch.softmax(network(**batch)[0].float(), dim=-1).tolist()

    if question["type"] == "choice":
        keys = list(question["criteria"])
        best_key = keys[max(range(len(keys)), key=probs.__getitem__)]
        return {"choice": best_key, "probabilities": dict(zip(keys, probs))}
    elif question["type"] == "score":
        expected_score = sum(i * p for i, p in enumerate(probs))
        return {"score": expected_score, "probabilities": probs}
    else:  # noul
        return {"p_yes": probs[1], "probabilities": {"No": probs[0], "Yes": probs[1]}}

# Load the trained checkpoint and run test cases
device = get_device()
eval_network, meta = load_checkpoint(final_ckpt, device)

print("=" * 60)
print("TEST 1: CHOICE (Multiple-Choice Decision)")
print("=" * 60)
sample_state_choice = (
    "A client with an e-commerce platform reports that during flash sales, the database connection pool "
    "becomes completely saturated, leading to 504 Gateway Timeouts. Read traffic is 95% of total volume."
)
sample_q_choice = {
    "type": "choice",
    "instructions": "Select the most effective immediate architectural remediation.",
    "criteria": {
        "A": "Add a Redis caching layer for read-heavy product catalog queries.",
        "B": "Migrate all databases from SQL to Cassandra overnight.",
        "C": "Restart the web servers every 10 minutes during the sale.",
    }
}
res_choice = answer(eval_network, meta, sample_state_choice, sample_q_choice, device)
print(json.dumps(res_choice, indent=2))

print("\\n" + "=" * 60)
print("TEST 2: NOUL (Binary Yes/No Proposition)")
print("=" * 60)
sample_state_noul = (
    "The commit modifies production encryption keys in plain text within a public repository."
)
sample_q_noul = {
    "type": "noul",
    "instructions": "Does this commit introduce a critical security vulnerability?",
}
res_noul = answer(eval_network, meta, sample_state_noul, sample_q_noul, device)
print(json.dumps(res_noul, indent=2))

print("\\n" + "=" * 60)
print("TEST 3: SCORE (Ordinal Scale 0 to 3)")
print("=" * 60)
sample_state_score = (
    "System metrics report CPU utilization at 98%, memory at 94%, and 5xx error rate at 12%."
)
sample_q_score = {
    "type": "score",
    "instructions": "Rate incident severity on a 4-point scale.",
    "criteria": ["0: Normal", "1: Low Severity", "2: Major Outage", "3: Catastrophic Outage"]
}
res_score = answer(eval_network, meta, sample_state_score, sample_q_score, device)
print(json.dumps(res_score, indent=2))
""")

# ==========================================
# 12. PUSH TO HUGGING FACE HUB
# ==========================================
md("""
## 12. Push Trained Model & ChoiceHead to Hugging Face Hub
Publishes the fine-tuned LoRA weights, ChoiceHead checkpoints, configuration, and an auto-generated model card directly to your Hugging Face account.

### How to Authenticate:
* Option 1: In Kaggle, go to **Add-ons** $\\rightarrow$ **Secrets**, add a secret named `HF_TOKEN` with your [Hugging Face Write Token](https://huggingface.co/settings/tokens).
* Option 2: Enter your token interactively when prompted by `notebook_login()`.
""")

code("""
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, login

# =====================================================================
# 1. Hugging Face Authentication
# =====================================================================
HF_TOKEN = None

# Attempt retrieval from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception:
    pass

if HF_TOKEN:
    login(token=HF_TOKEN)
    api = HfApi()
    user_info = api.whoami()
    hf_username = user_info["name"]
    repo_name = "qwen3-0.6b-jev-decision"
    REPO_ID = f"{hf_username}/{repo_name}"

    print(f"Target Hugging Face Repository: https://huggingface.co/{REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    ckpt_path = Path(final_ckpt)
    model_card = f\"\"\"---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
tags:
- decision-making
- text-classification
- multiple-choice
- bounded-decisions
- jev
- order-invariant
datasets:
- avbiswas/bev-decision
pipeline_tag: text-classification
---

# {repo_name}

This model is a **Judge / Evaluator / Verifier (JEV)** network fine-tuned from **Qwen3-0.6B** using choice-order invariant position encoding and custom 2D attention masking.

## Overview
- **Base Backbone:** Qwen3-0.6B (first 20 decoder layers)
- **Trained Components:** LoRA adapters (layers 8–19) + ChoiceHead attention scoring module
- **Dataset:** [`avbiswas/bev-decision`](https://huggingface.co/datasets/avbiswas/bev-decision)
- **Supported Decision Tasks:**
  - `CHOICE`: Multiple-choice option selection
  - `NOUL`: Binary Yes/No proposition ($P(\\\\text{{Yes}}))$)
  - `SCORE`: Ordinal grading on anchored discrete levels (0 to $N$)

## Checkpoint Contents
- `lora/`: PEFT adapter weights (`adapter_model.safetensors`, `adapter_config.json`)
- `head.pt`: State dictionary for the `ChoiceHead` attention module
- `meta.json`: Run metadata and layer configuration
- `config.yaml`: Complete hyperparameter recipe
\"\"\"

    (ckpt_path / "README.md").write_text(model_card, encoding="utf-8")
    print("Generated README.md model card.")

    print(f"Uploading checkpoint from {ckpt_path} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(ckpt_path),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Upload JEV trained checkpoint from Kaggle ({ckpt_path.name})"
    )
    print(f"\\nSuccessfully uploaded! View your model at:\\nhttps://huggingface.co/{REPO_ID}")
else:
    print("Notice: HF_TOKEN not configured in Kaggle Secrets.")
    print("Skipping automatic Hugging Face upload. All checkpoints are stored locally and zipped below.")
    print("To upload, add 'HF_TOKEN' under Kaggle Notebook -> Add-ons -> Secrets.")
""")

# ==========================================
# 13. EXPORT CHECKPOINTS (LOCAL ZIP)
# ==========================================
md("""
## 13. Export & Download Checkpoints (Local ZIP)
Packages all local run artifacts into a ZIP archive for direct download from Kaggle's `/kaggle/working` output pane.
""")

code("""
import os
!zip -rq /kaggle/working/jev_checkpoints.zip runs
print("Checkpoints packaged successfully to /kaggle/working/jev_checkpoints.zip")
!ls -lh /kaggle/working/jev_checkpoints.zip
""")

# Write out the notebook JSON
out = pathlib.Path(__file__).with_name("bev_train_kaggle.ipynb")
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Successfully generated standalone Kaggle notebook at: {out.resolve()}")
