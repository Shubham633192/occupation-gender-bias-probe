# Occupation–Gender Bias Probe (P49)
**Generative AI Course Project** | **Interactive Gradio Dashboard**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE_NOTES.md)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)

---

## 📌 Project Overview

The **Occupation–Gender Bias Probe (P49)** is a lightweight, responsive diagnostic app for measuring and visualizing occupational gender stereotyping across foundation models:
- **LLM 1**: `Qwen/Qwen2.5-0.5B-Instruct`
- **LLM 2**: `distilgpt2` (lightweight CPU-friendly LM)
- **Masked LM**: `bert-base-uncased`

Heavy benchmark results are pre-computed in `results/` so the app loads **instantly** without waiting for slow downloads or GPU requirements.

---

## 🚀 Quickstart & Run Command

### 1. Setup Environment
```bash
# Clone or navigate to the directory
cd occupation_gender_bias_probe

# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Launch App
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:7860
```

---

## 📱 Dashboard Tabs

| Tab | Name | Purpose |
| :---: | :--- | :--- |
| **1** | **Probe** | Real-time query interface: Enter any occupation (e.g. `nurse`, `engineer`) or sentence. View $P(\text{he})$ vs $P(\text{she})$, log-odds bias scores across LLM1, LLM2, and BERT, and trigger Responsible AI guardrails. |
| **2** | **Bias Chart** | Interactive Plotly grouped bar chart comparing all 10 occupations across 50 prompt templates for LLM1, LLM2, and BERT with reference neutrality line at 0. |
| **3** | **Mitigation** | Benchmark of baseline plain prompts vs balanced counter-stereotypical few-shot prompting, demonstrating an ~82.9% reduction in bias. |
| **4** | **About** | Explanations of bias propagation mechanisms (training data, tokenization, alignment), generative vs discriminative bias note, lifecycle diagram, and guardrail policies. |

---

## 📊 Summary of Bias Scores Across 10 Occupations

$$\text{Bias Score} = \ln\left(\frac{P(\text{he})}{P(\text{she})}\right)$$

| Occupation | Stereotype | LLM 1 (Qwen 0.5B) | LLM 2 (DistilGPT2) | BERT (bert-base) | Direction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nurse** | Female-Coded | **-1.84** | **-1.58** | **-2.15** | Female Skew |
| **Engineer** | Male-Coded | **+1.92** | **+1.74** | **+2.08** | Male Skew |
| **Doctor** | Male-Coded | **+1.41** | **+1.48** | **+1.72** | Male Skew |
| **CEO** | Male-Coded | **+2.14** | **+2.25** | **+2.45** | Male Skew |
| **Teacher** | Female-Coded | **-0.65** | **-0.39** | **-0.88** | Female Skew |
| **Secretary** | Female-Coded | **-1.48** | **-1.51** | **-1.94** | Female Skew |
| **Pilot** | Male-Coded | **+1.78** | **+1.86** | **+2.20** | Male Skew |
| **Librarian** | Female-Coded | **-1.12** | **-0.94** | **-1.35** | Female Skew |
| **Mechanic** | Male-Coded | **+2.35** | **+2.42** | **+2.68** | Male Skew |
| **Receptionist** | Female-Coded | **-1.72** | **-1.65** | **-2.02** | Female Skew |

---

## 🐙 Push to GitHub / Hugging Face Spaces

```bash
# Initialize git repository
git init
git add .
git commit -m "feat: complete occupation gender bias probe app P49"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/occupation-gender-bias-probe.git
git branch -M main
git push -u origin main

# Push to Hugging Face Spaces
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/occupation-gender-bias-probe
git push space main
```

---

## 📜 License
Distributed under the MIT License. See [LICENSE_NOTES.md](LICENSE_NOTES.md).
