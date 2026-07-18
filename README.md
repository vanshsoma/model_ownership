# Non-Fine-Tunable Model Protection (SOPHON-style)

Milestone 1: reproduce first-order non-fine-tunable learning on a small
classifier, and — just as important — **measure how easily it breaks**.

## Locked scope
- **Threat model:** white-box (attacker has the weights).
- **Goal:** category-C *non-fine-tunability* — make the released model
  resistant to being fine-tuned onto a restricted domain. This does **not**
  prevent copying and does **not** prove ownership; it degrades the usefulness
  of a fine-tuned copy.
- **Authorized domain:** CIFAR-10. **Restricted domain:** SVHN.
- **Method:** SOPHON's two alternating loops — Fine-Tuning Suppression (FTS)
  and Normal Training Reinforcement (NTR) — with a **first-order (FOMAML)**
  meta-gradient to keep compute single-GPU-friendly.

## The mechanism (see `sophon.py`)
- **FTS** clones the backbone, simulates `K` steps of an attacker fine-tuning
  on SVHN, then meta-updates the real backbone so that adapted predictions
  collapse toward **uniform** (`kl_to_uniform`, a *bounded* objective — this is
  why it converges where naive loss-maximization diverges).
- **NTR** interleaves ordinary CIFAR-10 training so protection doesn't wreck
  the shipped model.

## Run it
```bash
pip install -r requirements.txt

# clean baseline (warm-up only, no suppression)
python train.py --rounds 0    --out clean.pt

# SOPHON-protected model
python train.py --rounds 2000 --out protected.pt

# attack all three and compare SVHN recovery curves
python evaluate.py --mode scratch
python evaluate.py --mode load --ckpt clean.pt
python evaluate.py --mode load --ckpt protected.pt
```

## Run it on Colab (recommended — no local NVIDIA GPU)
This machine has no CUDA GPU, so real runs go on a free cloud T4.
1. New notebook at colab.research.google.com → **Runtime → Change runtime type → T4 GPU**.
2. Upload the 5 `.py` files (drag them into the file sidebar — multi-select works).
   torch/torchvision are preinstalled, so no `pip install` needed.
3. Run, GPU is picked up automatically:
   ```python
   !python train.py --warmup 50 --rounds 50 --out smoke.pt   # ~1-2 min smoke test
   !python train.py --rounds 0    --warmup 3000 --out clean.pt
   !python train.py --rounds 2000 --warmup 1000 --out protected.pt
   !python evaluate.py --mode scratch
   !python evaluate.py --mode load --ckpt clean.pt
   !python evaluate.py --mode load --ckpt protected.pt
   ```
CIFAR-10/SVHN download automatically on first run (fast on Colab).

## Win condition
`protected` fine-tunes to SVHN **no faster than `scratch`**, while `clean`
recovers SVHN accuracy quickly — and protected CIFAR-10 accuracy stays high.

## Milestone 2 (do this next — it's what makes the result credible)
Add adaptive attacks to `evaluate.py` and report protection as a function of
attacker effort: (a) **distillation** via query access, (b) **heal-then-finetune**
(warm on CIFAR first), (c) **layer-reset / linear-probe**, (d) **LoRA**.
A non-fine-tunability claim with no adaptive-attack evaluation isn't credible.

## Milestone 3 (the novel part)
Generalize suppression from one *named* restricted domain to *unseen* tasks:
train FTS over a diverse basket of proxy restricted domains and test whether
non-fine-tunability transfers to held-out ones.

## Honest caveats
- First-order approximation; hyperparameters (`K`, `meta_lr`, `inner_lr`,
  `ntr_per_round`) need tuning — suppression vs. authorized-accuracy is a
  genuine tension.
- A fresh attacker head each FTS round makes suppression robust to head init
  but weakens the per-round signal; raise `K` if suppression stalls.
- Not executed on hardware yet — this is a runnable starting point, not tuned
  results.
