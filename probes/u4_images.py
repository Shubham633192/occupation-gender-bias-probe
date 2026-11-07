"""
U4 - Image Generation & CLIP Tagging Audit Module
Generates images for occupations using sd-turbo (or HF Inference API / precomputed assets)
and tags perceived gender representation using CLIP (openai/clip-vit-base-patch32) zero-shot classification.
Compares visual representation percentages (% male vs % female) against text LLM bias scores.
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import pandas as pd


CLIP_CANDIDATE_LABELS = ["a photo of a man", "a photo of a woman"]


def tag_image_gender_clip(
    image,
    clip_model=None,
    clip_processor=None,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Performs zero-shot classification on an image using CLIP to determine
    perceived gender representation.
    """
    import torch
    from transformers import CLIPProcessor, CLIPModel

    if labels is None:
        labels = CLIP_CANDIDATE_LABELS

    if clip_model is None or clip_processor is None:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = clip_processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0].tolist()

    p_male = probs[0]
    p_female = probs[1]
    perceived_gender = "male" if p_male >= p_female else "female"

    return {
        "p_male": p_male,
        "p_female": p_female,
        "perceived_gender": perceived_gender,
        "confidence": max(p_male, p_female)
    }


def generate_image_sd_turbo(
    prompt: str,
    pipeline=None,
    hf_token: Optional[str] = None
):
    """
    Generates an image via diffusers SD-Turbo or HF Inference API.
    """
    # Check HF Inference API first if token is available
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token:
        try:
            import requests
            import io
            from PIL import Image
            api_url = "https://api-inference.huggingface.co/models/stabilityai/sd-turbo"
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.post(api_url, headers=headers, json={"inputs": prompt}, timeout=25)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"HF API image generation fallback: {e}")

    # Fallback to local diffusers pipeline
    from diffusers import AutoPipelineForText2Image
    import torch

    if pipeline is None:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float32
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")

    image = pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    return image


def load_or_compute_image_audit(audit_csv_path: str) -> pd.DataFrame:
    """
    Loads precomputed image audit data or returns existing DataFrame.
    """
    if os.path.exists(audit_csv_path):
        return pd.read_csv(audit_csv_path)
    raise FileNotFoundError(f"Audit file not found at {audit_csv_path}")
