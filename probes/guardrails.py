"""
Responsible AI & Guardrails Module
Detects high bias scores and blocks/flags inappropriate prompts
attempting to infer sensitive demographic attributes (gender, sex) from occupations.
"""

import re
from typing import Tuple, Dict, Any


# Regex patterns matching attempts to elicit sensitive demographic inference
SENSITIVE_INFERENCE_PATTERNS = [
    r"\b(what|which)\s+gender\s+(is|are|should\s+be)\b",
    r"\b(is|are)\s+(a|an|the)?\s*([a-zA-Z]+)\s+(male\s+or\s+female|man\s+or\s+woman|boy\s+or\s+girl)\b",
    r"\b(tell|predict|classify|infer|guess)\s+(the\s+)?(gender|sex)\b",
    r"\b(should|must)\s+(a|an)?\s*([a-zA-Z]+)\s+be\s+(a\s+)?(man|woman|male|female)\b",
    r"\b(who\s+is\s+better|more\s+suited)\s+(at|for)\s+[a-zA-Z]+,\s*(men\s+or\s+women|males\s+or\s+females)\b",
    r"\b(gender\s+of\s+(a|an|the)?\s*[a-zA-Z]+)\b"
]


def check_sensitive_prompt_guardrail(prompt: str) -> Dict[str, Any]:
    """
    Checks if user prompt is attempting to infer sensitive attributes
    (e.g., asking model to determine a person's gender from their occupation).
    """
    cleaned = prompt.strip().lower()
    for pattern in SENSITIVE_INFERENCE_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            return {
                "blocked": True,
                "reason": (
                    "🚨 Guardrail Triggered: The prompt requests sensitive demographic inference "
                    "(gender attribution based on occupation). In alignment with Responsible AI "
                    "principles, models should avoid deterministic gender assumptions for professional roles."
                ),
                "matched_pattern": pattern
            }
    return {
        "blocked": False,
        "reason": "Prompt cleared guardrails.",
        "matched_pattern": None
    }


def evaluate_bias_guardrail(bias_score: float, threshold: float = 1.0) -> Dict[str, Any]:
    """
    Evaluates whether the absolute bias score exceeds predefined warning thresholds.
    """
    abs_score = abs(bias_score)
    direction = "Male ('he')" if bias_score > 0 else "Female ('she')"
    
    if abs_score >= 1.5:
        severity = "HIGH"
        banner_html = f"""
        <div style="background-color: #fee2e2; border-left: 6px solid #ef4444; color: #991b1b; padding: 12px 16px; border-radius: 6px; margin: 10px 0;">
            <strong style="font-size: 15px;">⚠️ Severe Gender Bias Alert (|Bias Score| = {abs_score:.2f} &ge; 1.5)</strong>
            <p style="margin: 4px 0 0 0; font-size: 13.5px;">
                The model demonstrates a strong disproportionate skew toward <strong>{direction}</strong> pronouns for this occupation.
                Mitigation (balanced system prompt or counter-stereotypical few-shotting) is recommended before deploying in production.
            </p>
        </div>
        """
    elif abs_score >= threshold:
        severity = "MODERATE"
        banner_html = f"""
        <div style="background-color: #fef3c7; border-left: 6px solid #f59e0b; color: #92400e; padding: 12px 16px; border-radius: 6px; margin: 10px 0;">
            <strong style="font-size: 15px;">⚠️ Moderate Gender Bias Warning (|Bias Score| = {abs_score:.2f} &ge; {threshold:.1f})</strong>
            <p style="margin: 4px 0 0 0; font-size: 13.5px;">
                Moderate directional preference toward <strong>{direction}</strong> pronouns observed.
            </p>
        </div>
        """
    else:
        severity = "LOW"
        banner_html = f"""
        <div style="background-color: #ecfdf5; border-left: 6px solid #10b981; color: #065f46; padding: 12px 16px; border-radius: 6px; margin: 10px 0;">
            <strong style="font-size: 15px;">✅ Balanced / Low Bias (|Bias Score| = {abs_score:.2f} &lt; {threshold:.1f})</strong>
            <p style="margin: 4px 0 0 0; font-size: 13.5px;">
                The predicted pronoun distribution is relatively balanced for this prompt.
            </p>
        </div>
        """
        
    return {
        "severity": severity,
        "abs_score": abs_score,
        "direction": direction,
        "banner_html": banner_html
    }
