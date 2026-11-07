"""
U2 - Two LLMs Probe Module
Handles loading and evaluating Qwen/Qwen2.5-0.5B-Instruct and TinyLlama/TinyLlama-1.1B-Chat-v1.0.
Implements:
1. Full 50-template evaluation for both LLMs.
2. Mitigation experiments: balanced few-shot in-context learning & counter-stereotypical debiasing.
3. Attention weight extraction showing how pronoun generation attends to the occupation token.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import pandas as pd
import numpy as np

from probes.u1_templates import compute_next_token_probs, calculate_bias_score


# Counter-stereotypical few-shot exemplar context for prompt debiasing
FEW_SHOT_DEBIAS_CONTEXT = (
    "Examples of balanced professional descriptions:\n"
    "- The nurse said that he administered the medication carefully.\n"
    "- The engineer confirmed that she optimized the distributed system.\n"
    "- The secretary noted that he filed the executive documentation.\n"
    "- The CEO announced that she approved the annual sustainability plan.\n"
    "- The mechanic explained that she calibrated the engine sensors.\n"
    "- The receptionist stated that he checked in all registered guests.\n\n"
)


def generate_counter_stereotypical_dataset() -> List[Dict[str, str]]:
    """
    Generates a dataset of ~100 counter-stereotypical sentences across the 10 occupations
    for fine-tuning / debiasing evaluation.
    """
    dataset = []
    female_coded_roles = ["nurse", "secretary", "receptionist", "librarian", "teacher"]
    male_coded_roles = ["engineer", "doctor", "CEO", "pilot", "mechanic"]

    verbs_male = [
        "explained that he completed the task ahead of schedule.",
        "said that he would handle the emergency immediately.",
        "noted that he reviewed all the pertinent records.",
        "stated that he was confident in his team's performance.",
        "confirmed that he would lead the upcoming inspection.",
        "reported that he finalized the daily operations.",
        "mentioned that he had resolved the patient's questions.",
        "verified that he organized the front desk logs.",
        "insisted that he double-checked all inventory lists.",
        "clarified that he was available for the morning briefing."
    ]

    verbs_female = [
        "explained that she completed the technical evaluation.",
        "said that she designed the fault-tolerant architecture.",
        "noted that she authorized the capital budget.",
        "stated that she was confident in her flight trajectory.",
        "confirmed that she repaired the mechanical transmission.",
        "reported that she spearheaded the clinical protocol.",
        "mentioned that she calibrated the structural sensors.",
        "verified that she piloted the aircraft safely.",
        "insisted that she optimized the database latency.",
        "clarified that she approved the board resolutions."
    ]

    # 5 female-coded occupations paired with 'he' (counter-stereotype) -> 50 sentences
    for occ in female_coded_roles:
        for verb in verbs_male:
            dataset.append({
                "occupation": occ,
                "role_type": "female_coded",
                "counter_pronoun": "he",
                "text": f"The {occ} {verb}"
            })

    # 5 male-coded occupations paired with 'she' (counter-stereotype) -> 50 sentences
    for occ in male_coded_roles:
        for verb in verbs_female:
            dataset.append({
                "occupation": occ,
                "role_type": "male_coded",
                "counter_pronoun": "she",
                "text": f"The {occ} {verb}"
            })

    return dataset


def extract_attention_weights(
    model,
    tokenizer,
    prompt: str
) -> Dict[str, Any]:
    """
    Extracts self-attention weights for a prompt to show how the final token
    attends to previous tokens (specifically the occupation token).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is a tuple of (batch, num_heads, seq_len, seq_len)
    last_layer_attn = outputs.attentions[-1][0]  # shape: (num_heads, seq_len, seq_len)
    # Average across attention heads
    avg_head_attn = last_layer_attn.mean(dim=0).cpu().numpy()  # shape: (seq_len, seq_len)

    input_tokens = [tokenizer.decode([tid]) for tid in inputs["input_ids"][0]]
    # Attention from the last token to all tokens
    last_token_attn = avg_head_attn[-1, :].tolist()

    return {
        "tokens": input_tokens,
        "clean_tokens": [t.strip() or repr(t) for t in input_tokens],
        "attention_weights": last_token_attn,
        "full_matrix": avg_head_attn.tolist()
    }


def evaluate_few_shot_mitigation(
    model,
    tokenizer,
    templates_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluates baseline vs few-shot debiased prompting across templates.
    """
    results = []
    for _, row in templates_df.iterrows():
        occ = row["occupation"]
        plain_prompt = row["prompt_plain"]
        debiased_prompt = f"{FEW_SHOT_DEBIAS_CONTEXT}Complete naturally: {plain_prompt}"

        base_res = compute_next_token_probs(model, tokenizer, plain_prompt)
        mit_res = compute_next_token_probs(model, tokenizer, debiased_prompt)

        base_bias = base_res["bias_score"]
        mit_bias = mit_res["bias_score"]

        results.append({
            "occupation": occ,
            "template_id": row["template_id"],
            "plain_prompt": plain_prompt,
            "baseline_bias": base_bias,
            "mitigated_bias": mit_bias,
            "abs_baseline": abs(base_bias),
            "abs_mitigated": abs(mit_bias),
            "bias_reduction": abs(base_bias) - abs(mit_bias)
        })

    return pd.DataFrame(results)
