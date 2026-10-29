#!/usr/bin/env python3
import os, sys, math, json, time, argparse
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

class TextSequenceDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
        }

def load_task_data(task_name: str, tokenizer, max_len: int = 256, n_train: int = 2000, n_eval: int = 200):
    train_texts, eval_texts = [], []
    if task_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = [x["text"] for x in ds["train"] if len(x["text"].strip()) > 30][:n_train]
        eval_texts = [x["text"] for x in ds["validation"] if len(x["text"].strip()) > 30][:n_eval]
    elif task_name == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        train_texts = [f"Question: {x['question']}\nAnswer: {x['answer']}" for x in ds["train"]][:n_train]
        eval_texts = [f"Question: {x['question']}\nAnswer: {x['answer']}" for x in ds["test"]][:n_eval]
    elif task_name == "imdb":
        ds = load_dataset("imdb")
        train_texts = [x["text"] for x in ds["train"]][:n_train]
        eval_texts = [x["text"] for x in ds["test"]][:n_eval]
    elif task_name == "ag_news":
        ds = load_dataset("ag_news")
        train_texts = [x["text"] for x in ds["train"]][:n_train]
        eval_texts = [x["text"] for x in ds["test"]][:n_eval]
    else:
        raise ValueError(f"Unknown task: {task_name}")

    def tokenize_list(texts):
        return tokenizer(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors=None)

    return TextSequenceDataset(tokenize_list(train_texts)), TextSequenceDataset(tokenize_list(eval_texts))

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def get_delta_weight(self) -> torch.Tensor:
        return (self.scaling * (self.lora_B @ self.lora_A)).detach().float()

def apply_lora_to_model(model: nn.Module, target_modules: List[str], rank: int = 16, alpha: float = 32.0):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_modules):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))

def reset_lora_adapters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset_parameters()

class CovarianceAccumulator:
    def __init__(self, model: nn.Module, target_modules: List[str], device: str = "cuda"):
        self.model = model
        self.device = device
        self.covariances: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}
        self.hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear) or (isinstance(module, nn.Linear) and any(t in name for t in target_modules)):
                self.hooks.append(module.register_forward_hook(self._create_hook(name)))

    def _create_hook(self, layer_name: str):
        def hook(mod, inputs, outputs):
            x = inputs[0].detach().view(-1, inputs[0].size(-1)).float()
            cov_step = x.T @ x
            n_tokens = x.size(0)
            if layer_name not in self.covariances:
                self.covariances[layer_name] = cov_step.cpu()
                self.counts[layer_name] = n_tokens
            else:
                self.covariances[layer_name] += cov_step.cpu()
                self.counts[layer_name] += n_tokens
        return hook

    def collect(self, dataloader: DataLoader, device: str, amp: bool = True):
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=amp):
                    _ = self.model(input_ids=input_ids, attention_mask=attn_mask)

    def get_normalized_covariances(self) -> Dict[str, torch.Tensor]:
        return {k: self.covariances[k] / max(1, self.counts[k]) for k in self.covariances}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def sleep_merge(model: nn.Module, method: str, var_retention: Optional[float] = None, covariances: Optional[Dict[str, torch.Tensor]] = None, alpha_proj: float = 1.0):
    total_rank_frac, total_norm_kept = [], []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            dW = module.get_delta_weight()
            dW_orig_norm = torch.norm(dW).item()
            if method == "direct" or covariances is None or name not in covariances:
                dW_eff = dW
                rank_frac = 0.0
            elif method == "wsnsc":
                cov = covariances[name].to(dW.device)
                eigvals, eigvecs = torch.linalg.eigh(cov)
                idx = torch.argsort(eigvals, descending=True)
                eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
                cum_var = torch.cumsum(eigvals, dim=0) / torch.sum(eigvals)
                k = (cum_var >= var_retention).nonzero()
                k = k[0].item() + 1 if len(k) > 0 else cov.shape[0]
                Uk = eigvecs[:, :k]
                dW_eff = dW - (alpha_proj * ((dW @ Uk) @ Uk.T))
                rank_frac = k / cov.shape[0]
            dW_eff_norm = torch.norm(dW_eff).item()
            total_rank_frac.append(rank_frac)
            total_norm_kept.append(dW_eff_norm / max(1e-8, dW_orig_norm))
            module.base_layer.weight.data.add_(dW_eff.to(module.base_layer.weight.dtype))
    reset_lora_adapters(model)
    return float(sum(total_rank_frac) / max(1, len(total_rank_frac))), float(sum(total_norm_kept) / max(1, len(total_norm_kept)))

def make_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels

def train_lora(model: nn.Module, dataloader: DataLoader, args, device: str):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    step = 0
    iterator = iter(dataloader)
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = make_labels(input_ids, attn_mask)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % 40 == 0 or step == args.steps:
            print(f"      step {step:4d}/{args.steps}  train loss {loss.item():.4f}")

def evaluate_all_tasks(model: nn.Module, eval_loaders: Dict[str, DataLoader], device: str, amp: bool = True) -> Dict[str, float]:
    model.eval()
    losses = {}
    with torch.no_grad():
        for task_name, loader in eval_loaders.items():
            total_loss, total_tokens = 0.0, 0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = make_labels(input_ids, attn_mask)
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    loss = out.loss
                n_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
            losses[task_name] = total_loss / max(1, total_tokens)
    return losses

def run_sequence(label: str, method: str, var_retention: Optional[float], args, build_model_fn, tasks: List[str], anchor: str, train_loaders: Dict[str, DataLoader], eval_loaders: Dict[str, DataLoader], device: str) -> Dict:
    print(f"\n=== {label} ===")
    model = build_model_fn()
    base_losses = evaluate_all_tasks(model, eval_loaders, device, args.amp)
    print(f"  base losses: {base_losses}")
    record = {"label": label, "method": method, "var_retention": var_retention, "init": deepcopy(base_losses), "stages": []}
    accum = None
    if method == "wsnsc":
        print("  collecting anchor covariance ...")
        accum = CovarianceAccumulator(model, args.target_modules, device)
        accum.collect(train_loaders[anchor], device, args.amp)

    for day_idx, task in enumerate(tasks, 1):
        t0 = time.time()
        print(f"  [day {day_idx}/{len(tasks)}] WAKE on '{task}'")
        train_lora(model, train_loaders[task], args, device)
        wake_losses = evaluate_all_tasks(model, eval_loaders, device, args.amp)
        print(f"  [day {day_idx}] SLEEP ({'direct' if method == 'direct' else 'wsnsc'})")
        covs = accum.get_normalized_covariances() if accum is not None else None
        rank_frac, dW_kept = sleep_merge(model, method, var_retention, covs, getattr(args, "alpha_proj", 1.0))
        post_losses = evaluate_all_tasks(model, eval_loaders, device, args.amp)
        elapsed = time.time() - t0
        print(f"    {task}: wake {wake_losses[task]:.4f} -> post-sleep {post_losses[task]:.4f} | anchor {post_losses[anchor]:.4f} | rank_frac {rank_frac:.3f} | dW kept {dW_kept:.3f} | {int(elapsed)}s")
        if accum is not None:
            accum.collect(train_loaders[task], device, args.amp)
        record["stages"].append({"task": task, "wake": wake_losses, "post": post_losses, "merge": {"subspace_rank_frac": rank_frac, "delta_norm_kept": dW_kept}})
    if accum is not None:
        accum.remove_hooks()
    del model
    torch.cuda.empty_cache()
    return record

def main():
    parser = argparse.ArgumentParser(description="WS-NSC Continual Learning Benchmark")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "imdb", "ag_news"])
    parser.add_argument("--anchor", type=str, default="wikitext")
    parser.add_argument("--var-retention", nargs="+", type=float, default=[0.90, 0.95])
    parser.add_argument("--alpha-proj", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_task_names = [args.anchor] + args.tasks
    train_sets, eval_sets = {}, {}
    for t in all_task_names:
        print(f"loading '{t}' ...")
        tr, ev = load_task_data(t, tokenizer, args.max_len, args.n_train, args.n_eval)
        train_sets[t] = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
        eval_sets[t] = DataLoader(ev, batch_size=args.eval_batch_size, shuffle=False)

    def build_model():
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        apply_lora_to_model(model, args.target_modules, rank=args.rank, alpha=args.alpha)
        model.to(device)
        return model

    records = [run_sequence("direct-merge", "direct", None, args, build_model, args.tasks, args.anchor, train_sets, eval_sets, device)]
    for v in args.var_retention:
        records.append(run_sequence(f"ws-nsc@{v}", "wsnsc", v, args, build_model, args.tasks, args.anchor, train_sets, eval_sets, device))

    summary = []
    for r in records:
        lbl, init, final_post = r["label"], r["init"], r["stages"][-1]["post"]
        gains_wake = [(init[s["task"]] - s["wake"][s["task"]]) / init[s["task"]] * 100 for s in r["stages"]]
        gains_post = [(init[s["task"]] - s["post"][s["task"]]) / init[s["task"]] * 100 for s in r["stages"]]
        m_wake, m_post = sum(gains_wake)/len(gains_wake), sum(gains_post)/len(gains_post)
        forget_nats = [final_post[args.tasks[i]] - r["stages"][i]["post"][args.tasks[i]] for i in range(len(args.tasks)-1)]
        forget_rel = [(final_post[args.tasks[i]] - r["stages"][i]["post"][args.tasks[i]]) / r["stages"][i]["post"][args.tasks[i]] * 100 for i in range(len(args.tasks)-1)]
        summary.append({
            "label": lbl,
            "plasticity_gain_wake_%": m_wake,
            "plasticity_gain_post_%": m_post,
            "plasticity_retention": m_post / max(1e-8, m_wake),
            "avg_forgetting_nats": sum(forget_nats)/len(forget_nats),
            "avg_forgetting_rel_%": sum(forget_rel)/len(forget_rel),
            "anchor_drift_nats": final_post[args.anchor] - init[args.anchor],
            "rank_frac": sum(s["merge"]["subspace_rank_frac"] for s in r["stages"])/len(r["stages"]),
            "dW_kept": sum(s["merge"]["delta_norm_kept"] for s in r["stages"])/len(r["stages"])
        })

    print("\n" + "=" * 80)
    print(f"{'method':<18} {'gain-wake%':>11} {'gain-post%':>11} {'plast-ret':>11} {'forget':>8} {'forget%':>8} {'anchorΔ':>8} {'rank/d':>7} {'|dW|kept':>8}")
    print("-" * 80)
    for s in summary:
        print(f"{s['label']:<18} {s['plasticity_gain_wake_%']:11.3f} {s['plasticity_gain_post_%']:11.3f} {s['plasticity_retention']:11.3f} {s['avg_forgetting_nats']:8.3f} {s['avg_forgetting_rel_%']:8.3f} {s['anchor_drift_nats']:8.3f} {s['rank_frac']:7.3f} {s['dW_kept']:8.3f}")
    print("=" * 80)

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, "results.json")
    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "records": records, "summary": summary}, f, indent=2)
    print(f"\nSaved results -> {out_file}")

if __name__ == "__main__":
    main()
