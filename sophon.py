import copy
import math
import torch
import torch.nn.functional as F

from model import Head


def kl_to_uniform(logits):
    """KL(p || uniform) = log C - H(p). Minimizing it pushes predictions toward
    uniform noise; bounded below by 0 so the meta-loop stays stable."""
    C = logits.size(1)
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    return (p * (log_p + math.log(C))).sum(dim=1).mean()


def fts_step(backbone, rest_stream, device, K=30, inner_lr=0.01,
             meta_lr=5e-3, clip=10.0):
    """Fine-Tuning Suppression via first-order MAML."""
    fast_bb = copy.deepcopy(backbone)
    fast_head = Head().to(device)          # fresh head => robust to attacker head init
    fast_params = list(fast_bb.parameters()) + list(fast_head.parameters())
    # Momentum matches the REAL attacker in evaluate.py. Without it the simulated
    # attacker converges far slower (~40% vs ~80% in the same steps), so its
    # predictions never get confident and the suppression gradient stays weak.
    inner_opt = torch.optim.SGD(fast_params, lr=inner_lr, momentum=0.9)

    # simulate the attacker minimizing CE on the restricted domain
    for _ in range(K):
        x, y = next(rest_stream)
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(fast_head(fast_bb(x)), y)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()

    # suppression objective at the adapted point
    x, y = next(rest_stream)
    x, y = x.to(device), y.to(device)
    logits = fast_head(fast_bb(x))
    adapted_acc = (logits.argmax(1) == y).float().mean().item()  # diagnostic
    l_sup = kl_to_uniform(logits)
    fast_bb.zero_grad(set_to_none=True)
    fast_head.zero_grad(set_to_none=True)
    l_sup.backward()

    # FOMAML meta-step: apply adapted-point grads to the real backbone
    with torch.no_grad():
        for real_p, fast_p in zip(backbone.parameters(), fast_bb.parameters()):
            if fast_p.grad is None:
                continue
            g = fast_p.grad
            if clip is not None:
                g = torch.clamp(g, -clip, clip)
            real_p -= meta_lr * g
    return l_sup.item(), adapted_acc


def ntr_step(backbone, auth_head, auth_opt, auth_stream, device):
    """Normal Training Reinforcement: one CE step on the authorized task."""
    x, y = next(auth_stream)
    x, y = x.to(device), y.to(device)
    loss = F.cross_entropy(auth_head(backbone(x)), y)
    auth_opt.zero_grad()
    loss.backward()
    auth_opt.step()
    return loss.item()
