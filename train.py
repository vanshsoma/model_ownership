import argparse
import torch

from data import cifar10_loaders, svhn_loaders, infinite
from model import build_backbone, Head
from sophon import fts_step, ntr_step


@torch.no_grad()
def accuracy(backbone, head, loader, device, max_batches=None):
    was_training = backbone.training
    backbone.eval()
    head.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        pred = head(backbone(x)).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    if was_training:
        backbone.train()
        head.train()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2000,
                    help="FTS/NTR rounds. Use 0 to produce the CLEAN baseline "
                         "(warm-up only, no suppression).")
    ap.add_argument("--warmup", type=int, default=1000,
                    help="NTR-only steps first, so there is a real authorized "
                         "model to protect.")
    ap.add_argument("--K", type=int, default=30, help="simulated attacker steps")
    ap.add_argument("--inner_lr", type=float, default=0.01)
    ap.add_argument("--meta_lr", type=float, default=5e-3)
    ap.add_argument("--ntr_lr", type=float, default=0.01)
    ap.add_argument("--ntr_per_round", type=int, default=1)
    ap.add_argument("--out", default="protected.pt")
    ap.add_argument("--save_every", type=int, default=500,
                    help="checkpoint every N rounds so a Colab disconnect does "
                         "not lose the whole run")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    backbone = build_backbone().to(device)
    auth_head = Head().to(device)

    _, auth_test = cifar10_loaders()
    auth_train, _ = cifar10_loaders()
    rest_train, _ = svhn_loaders()
    auth_stream = infinite(auth_train)
    rest_stream = infinite(rest_train)

    auth_opt = torch.optim.SGD(
        list(backbone.parameters()) + list(auth_head.parameters()),
        lr=args.ntr_lr, momentum=0.9)

    print("Warm-up (authorized task only)...")
    for _ in range(args.warmup):
        ntr_step(backbone, auth_head, auth_opt, auth_stream, device)
    print(f"  CIFAR-10 acc after warm-up: "
          f"{accuracy(backbone, auth_head, auth_test, device, 20):.3f}")

    print("Alternating FTS / NTR...")
    for r in range(args.rounds):
        l_sup, sim_acc = fts_step(backbone, rest_stream, device,
                                  K=args.K, inner_lr=args.inner_lr, meta_lr=args.meta_lr)
        for _ in range(args.ntr_per_round):
            ntr_step(backbone, auth_head, auth_opt, auth_stream, device)
        if (r + 1) % 100 == 0:
            acc = accuracy(backbone, auth_head, auth_test, device, 20)
            print(f"round {r + 1:5d} | L_sup {l_sup:.4f} | "
                  f"sim-attacker SVHN {sim_acc:.3f} | CIFAR-10 acc {acc:.3f}")
        if (r + 1) % args.save_every == 0:
            torch.save({"backbone": backbone.state_dict(),
                        "auth_head": auth_head.state_dict()}, args.out)
            print(f"  checkpoint -> {args.out} (round {r + 1})")

    torch.save({"backbone": backbone.state_dict(),
                "auth_head": auth_head.state_dict()}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
