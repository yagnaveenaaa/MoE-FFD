import sys
path = sys.argv[1] if len(sys.argv) > 1 else "ViT_MoE.py"
src = open(path, encoding="utf-8").read()
if "GATE_IN" in src:
    sys.exit("Already patched (found GATE_IN). Nothing done.")
edits = [
 ("import math\n", "import math\nimport os\n"),
 ("        self.w_gate = nn.Parameter(torch.zeros(dim,  self.num_experts), requires_grad=True)\n"
  "        self.w_noise = nn.Parameter(torch.zeros(dim,  self.num_experts), requires_grad=True)\n",
  "        self.gate_in = os.environ.get(\"GATE_IN\", \"mean\")   # \"mean\" = baseline, \"cls_avg\" = proposed\n"
  "        assert self.gate_in in (\"mean\", \"cls_avg\")\n"
  "        gate_dim = 2 * dim if self.gate_in == \"cls_avg\" else dim\n"
  "        self.w_gate = nn.Parameter(torch.zeros(gate_dim, self.num_experts), requires_grad=True)\n"
  "        self.w_noise = nn.Parameter(torch.zeros(gate_dim, self.num_experts), requires_grad=True)\n"),
 ("        x_global = torch.mean(x,dim=1,keepdim=False)\n",
  "        if self.gate_in == \"cls_avg\":\n"
  "            x_global = torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=1)   # [B, 2*dim]\n"
  "        else:\n"
  "            x_global = torch.mean(x, dim=1, keepdim=False)                 # baseline, unchanged\n"),
 ("        gates, load = self.noisy_top_k_gating(x_global, self.training)\n",
  "        gates, load = self.noisy_top_k_gating(x_global, self.training)\n"
  "        self.last_gates = gates.detach()   # only used by routing_stats.py\n"),
]
for old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"ANCHOR PROBLEM: expected exactly 1 match, found {n} for:\n{old}")
    src = src.replace(old, new)
open(path, "w", encoding="utf-8").write(src)
print("Patched OK:", path)
