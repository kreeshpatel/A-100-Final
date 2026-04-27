# AI 100 Final Project — AI System Building with GenAI as a Cognitive Partner

**Author:** Kreesh Patel
**Course:** AI 100 (Spring 2026)
**LLM used:** ChatGPT
**Submission type:** Solo
**Base AI system:** MNIST CNN classifier from the midterm
([github.com/kreeshpatel/ai100-mnist-cnn-classification](https://github.com/kreeshpatel/ai100-mnist-cnn-classification))

## What this repo contains

Per the assignment spec, this submission includes a Google Sheet/Excel file
of "Bad" cases, a PDF reflection report, and runnable code files.

| Deliverable | File |
|---|---|
| Excel of 10 "Bad" cases | [`final_project_bad_cases.xlsx`](final_project_bad_cases.xlsx) |
| Reflection report (PDF) | [`final_project_report.pdf`](final_project_report.pdf) |
| Baseline midterm code | [`src/`](src/) |
| 10 buggy variants | [`bug_cases/`](bug_cases/) |

## The 10 bug cases

Each case is a single-edit change to `src/model.py` or `src/train.py`. The
files in `bug_cases/` are full drop-in replacements with a docstring header
that explains the modification, the expected failure, and how to reproduce.

| # | File touched | Bug | Failure mode |
|---|---|---|---|
| 1 | `model.py` | `Linear(64*7*7, 128)` → `Linear(64*14*14, 128)` | `RuntimeError`: shape mismatch on first forward |
| 2 | `model.py` | `Linear(128, 10)` → `Linear(128, 9)` | `IndexError`: target 9 out of bounds |
| 3 | `model.py` | `Conv2d(in_channels=1)` → `Conv2d(in_channels=3)` | `RuntimeError`: 3 channels expected, got 1 |
| 4 | `train.py` | `transforms.ToTensor()` removed | `TypeError`: PIL image fed to `Normalize` |
| 5 | `train.py` | `optimizer.zero_grad()` removed | Silent: training unstable, accuracy stalls |
| 6 | `train.py` | `torch.max(outputs, 1)` → `torch.max(outputs, 0)` | Silent: ~10% test accuracy |
| 7 | `train.py` | `CrossEntropyLoss()` → `NLLLoss()` | NaN loss, no convergence |
| 8 | `train.py` | `--lr` default `0.001` → `10.0` | Loss explodes to NaN |
| 9 | `train.py` | `optimizer.step()` and `loss.backward()` swapped | Silent: model never updates |
| 10 | `train.py` | Train loader `shuffle=True` → `shuffle=False` | Silent: a few points off baseline |

A mix of architecture, data-pipeline, training-loop, and config bugs;
roughly half crash and half fail silently.

## How to reproduce a bug

```bash
# Pick a case and copy it over the matching baseline file.
cp bug_cases/case_06_wrong_argmax_dim__train.py src/train.py

# Then run training.
python src/train.py
```

To restore the baseline, regenerate `src/train.py` (or `src/model.py`) from
this repo or from the midterm repo linked above.

## Setup

```bash
# from the project root
python -m venv .venv
.venv\Scripts\activate.ps1     # Windows PowerShell
pip install -r requirements.txt

# sanity check
python src/sanity_check.py

# baseline run (~98.8% test accuracy)
python src/train.py
```

