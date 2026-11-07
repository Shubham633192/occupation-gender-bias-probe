# License & Intellectual Property Notes

This document summarizes the licensing terms, redistribution rights, and intellectual property (IP) guidance for all models, libraries, and generative outputs utilized in the **Occupation–Gender Bias Probe (P49)** project.

---

## 1. Models & License Inventory

| Component | Model Identifier / Artifact | License Type | Commercial Use Allowed | Primary Source / Repo |
| :--- | :--- | :--- | :--- | :--- |
| **LLM 1** | `Qwen/Qwen2.5-0.5B-Instruct` | **Apache 2.0** | Yes (Permissive) | [Qwen Team / Alibaba](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| **LLM 2** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | **Apache 2.0** | Yes (Permissive) | [TinyLlama Team](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| **Masked LM** | `bert-base-uncased` | **Apache 2.0** | Yes (Permissive) | [Google / Hugging Face](https://huggingface.co/bert-base-uncased) |
| **Diffusion** | `stabilityai/sd-turbo` | **Stability AI Community License / Research** | Non-Commercial / Research | [Stability AI](https://huggingface.co/stabilityai/sd-turbo) |
| **Vision-Text** | `openai/clip-vit-base-patch32` | **MIT License** | Yes (Permissive) | [OpenAI](https://huggingface.co/openai/clip-vit-base-patch32) |
| **Application** | Occupation–Gender Bias Probe Codebase | **MIT License** | Yes | Individual Student Project |

---

## 2. Permissive Model Licences (Apache 2.0 & MIT)

- **Qwen2.5-0.5B-Instruct**, **TinyLlama-1.1B-Chat-v1.0**, and **bert-base-uncased** are distributed under the **Apache License 2.0**.
  - Permitted: Reproduction, modification, distribution, sublicensing, and commercial usage.
  - Requirement: Retain original copyright notices, disclaimer of warranty, and Apache 2.0 license text.
- **CLIP (openai/clip-vit-base-patch32)** is distributed under the **MIT License**.
  - Permitted: Free use, modification, distribution, and private/commercial deployment with copyright attribution.

---

## 3. Stability AI SD-Turbo Licensing Constraints

- `stabilityai/sd-turbo` is governed by the Stability AI Community License.
- **Academic / Educational Exemption**: This project is developed strictly for educational research within a Generative AI curriculum (Course Project P49).
- **Prohibited Uses**: Any unauthorized commercial distribution, high-risk automated decision making, or impersonation of living individuals without consent.

---

## 4. Generative AI Output & Intellectual Property (IP) Guidance

### A. Authorship & Copyrightability
- In accordance with the **U.S. Copyright Office (USCO) Guidance on AI-Generated Works (2023)** and international IP precedent:
  - Raw images generated purely from text prompts (e.g., `"a photo of a doctor"`) lack human authorship and are generally placed in the public domain.
  - The human compilation, scientific audit methodology, statistical probing pipeline, evaluation scripts, and analytical synthesis constitute copyrightable academic work authored by the researcher.

### B. Image Scraping & Fair Use Context
- The synthetic images generated in this project serve solely as probe targets to audit algorithmic representation across gender categories.
- No copyrighted third-party portraits or private biometrics are stored or reproduced.

---

## 5. Compliance Checklist for Hugging Face Spaces Deployment

- [x] Pre-computed image audit results ensure zero live GPU overhead on free Spaces tiers.
- [x] No proprietary dataset redistributions violate licensing agreements.
- [x] All Hugging Face model cards cited with direct upstream hyperlinks.
- [x] Sensitive demographic inference prompts blocked via guardrails.
