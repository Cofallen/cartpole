import torch
import argparse, os
from pathlib import Path

# policy = torch.jit.load("logs/rsl_rl/cartpole_direct/2026-11-02_18-48-52/exported/policy.pt").eval()
# obs = torch.tensor(([[0.1, 0.0, 0.0, 0.0]]), dtype=torch.float32)
# action = policy(obs)
# print(action)
# print(action.shape)


def load_policy(path):
    return torch.jit.load(path, map_location="cpu").eval()


def act(policy, obs):
    with torch.no_grad():
        x = torch.tensor([obs], dtype=torch.float32)
        return float(policy(x).reshape(-1)[0])


def main():
    policy = load_policy("logs/rsl_rl/cartpole_direct/2026-11-02_18-48-52/exported/policy.pt")
    obs = [0.1, 0.0, 0.0, 0.0]
    action = act(policy, obs)
    print(action)


if __name__ == "__main__":
    main()
