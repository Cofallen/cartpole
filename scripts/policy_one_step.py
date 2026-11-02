import torch
p = torch.jit.load("logs/rsl_rl/cartpole_direct/2026-11-02_18-48-52/exported/policy.pt").eval()
print(p(torch.tensor(([0.1, 0.0, 0.0, 0.0]))))