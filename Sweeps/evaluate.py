"""Private subprocess entry point: evaluate one explicit checkpoint to JSON."""

import argparse
from importlib import import_module
import random

import numpy as np
import torch

from Custom_enviornments.Test_Env.Env_16 import Env16
from Sweeps.runner import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=("ppo", "ppo_lstm"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    # Controls policy sampling only. Env16 does not seed the external game.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluator = import_module(f"Inference.{args.algorithm}_eval")
    device = torch.device("cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
    env = Env16()
    try:
        model = evaluator.Agent(env).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        returns, wins = evaluator.evaluate(env, model, args.episodes, device, not args.stochastic)
        write_json(args.output, {"episode_returns": returns, "wins": wins,
                                 "evaluation_seed": args.seed, "deterministic": not args.stochastic})
    finally:
        env.close()


if __name__ == "__main__":
    main()
