"""The attack harness. This is the part that decides whether protection worked.

We play the white-box attacker: take a backbone, attach a fresh head, and
fine-tune the whole thing on the restricted domain (SVHN). We report SVHN test
accuracy vs. fine-tuning steps.

Run three curves and compare:
    python train.py --rounds 0   --out clean.pt          # normally-pretrained
    python train.py --rounds 2000 --out protected.pt     # SOPHON-protected
    python evaluate.py --mode scratch                    # random init
    python evaluate.py --mode load --ckpt clean.pt
    python evaluate.py --mode load --ckpt protected.pt

WIN CONDITION: the protected curve tracks the `scratch` curve (fine-tuning the
protected model is no cheaper than training from scratch), while the `clean`
curve reaches high SVHN accuracy fast. Meanwhile protected CIFAR-10 accuracy
(printed by train.py) stays high.
"""
import argparse
import torch
import torch.nn.functional as F

from data import svhn_loaders, infinite
from model import build_backbone, Head
from train import accuracy


def finetune_attack(backbone, device, steps, lr, tag, limit=None):
    rest_train, rest_test = svhn_loaders(limit=limit)
    rest_stream = infinite(rest_train)
    head = Head().to(device)
    opt = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()),
                          lr=lr, momentum=0.9)
    backbone.train()
    head.train()
    for s in range(steps):
        x, y = next(rest_stream)
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(head(backbone(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 100 == 0:
            acc = accuracy(backbone, head, rest_test, device, 20)
            print(f"[{tag}] step {s + 1:5d} | SVHN acc {acc:.3f}")
    return accuracy(backbone, head, rest_test, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["load", "scratch"], default="load")
    ap.add_argument("--ckpt", default="protected.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None,
                    help="restrict the attacker to N SVHN samples (few-shot regime)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone().to(device)
    if args.mode == "load":
        state = torch.load(args.ckpt, map_location=device)
        backbone.load_state_dict(state["backbone"])
        tag = args.ckpt
    else:
        tag = "scratch"  # leave randomly initialized

    if args.limit is not None:
        tag = f"{tag}|{args.limit}-shot"
    final = finetune_attack(backbone, device, args.steps, args.lr, tag, limit=args.limit)
    print(f"FINAL SVHN acc [{tag}] after {args.steps} steps: {final:.3f}")


if __name__ == "__main__":
    main()
