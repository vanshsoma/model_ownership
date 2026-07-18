import copy
import math
import torch
import torch.nn.functional as F

from model import Head


def kl_to_uniform(logits):
    """KL(p || uniform) = log C - H(p).

    Bounded below by 0 (reached when p is uniform). We MINIMIZE this as the
    suppression target, which pushes the adapted model's predictions toward
    uniform noise. Unlike loss-maximization it is bounded and its gradient
    shrinks as suppression succeeds, so the meta-loop stays stable.
    """
    C = logits.size(1)
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    return (p * (log_p + math.log(C))).sum(dim=1).mean()


def fts_step(backbone, rest_stream, device, K=5, inner_lr=0.01,
             meta_lr=1e-3, clip=10.0):
    """Fine-Tuning Suppression, first-order MAML (FOMAML).

    1. Clone the backbone and attach a FRESH head  -> simulated attacker.
    2. Inner loop: K steps of ordinary CE fine-tuning on restricted data
       (exactly what the thief does).
    3. Evaluate the suppression objective at that ADAPTED point.
    4. Transplant the adapted-point gradients back onto the REAL backbone
       (the first-order approximation: we skip the 2nd-order term through
       the inner loop). This nudges the real weights so that the attacker's
       fine-tuning trajectory lands in a useless (uniform) region.
    """
    fast_bb = copy.deepcopy(backbone)
    fast_head = Head().to(device)          # fresh head => robust to attacker's head init
    fast_params = list(fast_bb.parameters()) + list(fast_head.parameters())
    inner_opt = torch.optim.SGD(fast_params, lr=inner_lr)

    # (2) simulate the attacker minimizing CE on the restricted domain
    for _ in range(K):
        x, y = next(rest_stream)
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(fast_head(fast_bb(x)), y)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()

    # (3) suppression objective at the adapted point
    x, _ = next(rest_stream)
    x = x.to(device)
    l_sup = kl_to_uniform(fast_head(fast_bb(x)))
    fast_bb.zero_grad(set_to_none=True)
    fast_head.zero_grad(set_to_none=True)
    l_sup.backward()

    # (4) FOMAML meta-step: apply adapted-point grads to the real backbone
    with torch.no_grad():
        for real_p, fast_p in zip(backbone.parameters(), fast_bb.parameters()):
            if fast_p.grad is None:
                continue
            g = fast_p.grad
            if clip is not None:
                g = torch.clamp(g, -clip, clip)
            real_p -= meta_lr * g
    return l_sup.item()


def ntr_step(backbone, auth_head, auth_opt, auth_stream, device):
    """Normal Training Reinforcement: one ordinary CE step on the authorized
    task, so suppression does not quietly destroy the model we ship."""
    x, y = next(auth_stream)
    x, y = x.to(device), y.to(device)
    loss = F.cross_entropy(auth_head(backbone(x)), y)
    auth_opt.zero_grad()
    loss.backward()
    auth_opt.step()
    return loss.item()
