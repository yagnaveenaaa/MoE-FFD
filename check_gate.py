import os, torch, torch.nn as nn
from ViT_MoE import vit_base_patch16_224_in21k
dev = "cuda" if torch.cuda.is_available() else "cpu"

def build(mode, seed=0):
    os.environ["GATE_IN"] = mode
    torch.manual_seed(seed)
    return vit_base_patch16_224_in21k(pretrained=False, num_classes=2).to(dev)

m0, m1 = build("mean"), build("cls_avg")
n0 = sum(p.numel() for p in m0.parameters() if p.requires_grad)
n1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
print("CHECK 1 trainable baseline:", n0, "| proposed:", n1, "| diff:", n1 - n0, "(expect 92160)")

sd0, sd1 = m0.state_dict(), m1.state_dict()
for k in sd1:
    if not k.endswith(("w_gate", "w_noise")):
        sd1[k] = sd0[k]
m1.load_state_dict(sd1)
x = torch.randn(4, 3, 224, 224, device=dev)
outs = []
for m in (m0, m1):
    m.train(); torch.manual_seed(123)
    with torch.no_grad():
        outs.append(m(x)[0])
print("CHECK 2 step-0 outputs identical:", torch.allclose(outs[0], outs[1], atol=1e-5))

y = torch.tensor([0, 1, 0, 1], device=dev)
for tag, use_moe in [("CE only", False), ("CE + moe_loss", True)]:
    m1.zero_grad(); m1.train(); torch.manual_seed(123)
    logits, moe = m1(x)
    loss = nn.functional.cross_entropy(logits, y) + (moe if use_moe else 0)
    loss.backward()
    g = [(n, None if p.grad is None else round(p.grad.abs().sum().item(), 6))
         for n, p in m1.named_parameters() if "adapter_MoE" in n and n.endswith(("w_gate", "w_noise"))]
    print("CHECK 3", tag, "| adapter-gate grads (first 4):", g[:4])
