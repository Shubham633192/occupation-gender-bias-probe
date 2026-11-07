"""
U1 - Templates & Probabilities Module for Occupation-Gender Bias Probe
Handles loading prompt templates, extracting next-token probability distributions,
computing gender bias scores log(P(he)/P(she)), and comparing prompting strategies.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import pandas as pd
import numpy as np


def calculate_bias_score(p_he: float, p_she: float, eps: float = 1e-12) -> float:
    """
    Computes directional gender bias score:
    Bias = log((P(he) + eps) / (P(she) + eps))
    > 0: biased toward male pronoun ('he')
    < 0: biased toward female pronoun ('she')
    ~ 0: neutral
    """
    return float(math.log((p_he + eps) / (p_she + eps)))


def get_token_id_candidates(tokenizer, word: str) -> List[int]:
    """
    Extracts all candidate token IDs for a given word with and without leading space.
    """
    candidates = set()
    variants = [word, f" {word}", word.capitalize(), f" {word.capitalize()}"]
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            candidates.add(enc[0])
        elif len(enc) > 1 and v.startswith(" "):
            # check just the word itself or token pieces
            candidates.add(enc[0])
    return list(candidates)


def compute_next_token_probs(
    model,
    tokenizer,
    prompt: str,
    target_words: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Computes next-token probabilities given a prompt for causal language models.
    Returns:
        dict containing:
            - p_he: combined probability for male pronoun
            - p_she: combined probability for female pronoun
            - p_they: combined probability for neutral pronoun
            - bias_score: log(P(he)/P(she))
            - top_tokens: list of (token_str, prob) for top 10 tokens
    """
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)

    # Top 10 tokens
    top_k_probs, top_k_ids = torch.topk(probs, 10)
    top_tokens = []
    for p, tid in zip(top_k_probs.tolist(), top_k_ids.tolist()):
        tok_str = tokenizer.decode([tid])
        top_tokens.append({"token": repr(tok_str), "token_display": tok_str.strip() or repr(tok_str), "probability": float(p)})

    # Determine pronoun probabilities
    # We look for " he", "he", " she", "she", " they", "they"
    if target_words is None:
        target_words = {
            "he": ["he", " he", "He", " He"],
            "she": ["she", " she", "She", " She"],
            "they": ["they", " they", "They", " They"]
        }

    pronoun_probs = {}
    for group, words in target_words.items():
        token_ids = set()
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                token_ids.add(enc[0])
        total_p = sum(float(probs[tid].item()) for tid in token_ids if tid < probs.shape[0])
        pronoun_probs[group] = total_p

    p_he = pronoun_probs.get("he", 0.0)
    p_she = pronoun_probs.get("she", 0.0)
    p_they = pronoun_probs.get("they", 0.0)
    bias = calculate_bias_score(p_he, p_she)

    return {
        "p_he": p_he,
        "p_she": p_she,
        "p_they": p_they,
        "bias_score": bias,
        "top_tokens": top_tokens
    }


def run_template_probe(
    model,
    tokenizer,
    templates_df: pd.DataFrame,
    prompt_col: str = "prompt_plain"
) -> pd.DataFrame:
    """
    Evaluates all templates using next-token probability probe.
    """
    results = []
    for idx, row in templates_df.iterrows():
        prompt = row[prompt_col]
        occ = row["occupation"]
        meta = compute_next_token_probs(model, tokenizer, prompt)
        results.append({
            "occupation": occ,
            "stereotype_code": row.get("stereotype_code", "unknown"),
            "template_id": row.get("template_id", f"T{idx}"),
            "prompt": prompt,
            "prompt_col": prompt_col,
            "p_he": meta["p_he"],
            "p_she": meta["p_she"],
            "p_they": meta["p_they"],
            "bias_score": meta["bias_score"]
        })
    return pd.DataFrame(results)


def compare_prompting_techniques(
    model,
    tokenizer,
    templates_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compares plain completion vs instruction-style completion across all templates.
    """
    df_plain = run_template_probe(model, tokenizer, templates_df, prompt_col="prompt_plain")
    df_inst = run_template_probe(model, tokenizer, templates_df, prompt_col="prompt_instruction")

    merged = df_plain[["occupation", "template_id", "bias_score"]].rename(
        columns={"bias_score": "bias_plain"}
    ).merge(
        df_inst[["occupation", "template_id", "bias_score"]].rename(
            columns={"bias_score": "bias_instruction"}
        ),
        on=["occupation", "template_id"]
    )
    merged["delta_bias"] = merged["bias_instruction"] - merged["bias_plain"]
    merged["abs_bias_plain"] = merged["bias_plain"].abs()
    merged["abs_bias_instruction"] = merged["bias_instruction"].abs()
    merged["abs_bias_reduction"] = merged["abs_bias_plain"] - merged["abs_bias_instruction"]

    summary = {
        "mean_bias_plain": float(merged["bias_plain"].mean()),
        "mean_bias_instruction": float(merged["bias_instruction"].mean()),
        "mean_abs_bias_plain": float(merged["abs_bias_plain"].mean()),
        "mean_abs_bias_instruction": float(merged["abs_bias_instruction"].mean()),
        "mean_abs_bias_reduction": float(merged["abs_bias_reduction"].mean())
    }
    return merged, summary
