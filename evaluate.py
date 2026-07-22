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

from data import domain_loaders, cifar10_loaders, infinite
from model import build_backbone, Head
from train import accuracy


def _make_opt(params, lr, optimizer):
    """Adam is an ADAPTIVE attack: the defense only ever simulated SGD."""
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr, momentum=0.9)


def heal(backbone, device, steps, lr):
    """Adaptive attack: warm the model on the AUTHORIZED domain first, dragging
    it off the fragile suppressed point, and only then attack the restricted
    domain. This is the classic way to defeat a fine-tuning-based defense."""
    auth_train, _ = cifar10_loaders()
    stream = infinite(auth_train)
    head = Head().to(device)
    opt = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()),
                          lr=lr, momentum=0.9)
    backbone.train()
    head.train()
    for _ in range(steps):
        x, y = next(stream)
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(head(backbone(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return backbone


def finetune_attack(backbone, device, steps, lr, tag, limit=None, optimizer="sgd",
                    domain="svhn"):
    rest_train, rest_test = domain_loaders(domain, limit=limit)
    rest_stream = infinite(rest_train)
    head = Head().to(device)
    opt = _make_opt(list(backbone.parameters()) + list(head.parameters()), lr, optimizer)
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
            print(f"[{tag}] step {s + 1:5d} | acc {acc:.3f}")
    return accuracy(backbone, head, rest_test, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["load", "scratch"], default="load")
    ap.add_argument("--ckpt", default="protected.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None,
                    help="restrict the attacker to N samples (few-shot regime)")
    ap.add_argument("--domain", default="svhn",
                    help="restricted domain to attack; use a HELD-OUT domain "
                         "(e.g. kmnist) to test unseen-domain generalization")
    ap.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd",
                    help="adaptive attack: the defense only simulated SGD")
    ap.add_argument("--heal", type=int, default=0,
                    help="adaptive attack: N warm-up steps on CIFAR-10 before "
                         "attacking, to escape the suppressed point")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone().to(device)
    if args.mode == "load":
        state = torch.load(args.ckpt, map_location=device)
        backbone.load_state_dict(state["backbone"])
        tag = args.ckpt
    else:
        tag = "scratch"  # leave randomly initialized

    tag = f"{tag}|{args.domain}"
    if args.limit is not None:
        tag = f"{tag}|{args.limit}-shot"
    if args.heal:
        tag = f"{tag}|heal{args.heal}"
        heal(backbone, device, args.heal, args.lr)
    if args.optimizer != "sgd":
        tag = f"{tag}|{args.optimizer}"
    final = finetune_attack(backbone, device, args.steps, args.lr, tag,
                            limit=args.limit, optimizer=args.optimizer, domain=args.domain)
    print(f"FINAL SVHN acc [{tag}] after {args.steps} steps: {final:.3f}")


if __name__ == "__main__":
    main()
