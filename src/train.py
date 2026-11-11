"""
Train the transformer language model on a text file.
Usage: python train.py [--text path] [--steps N] [--config small|medium|large]
"""

import argparse
import os
import numpy as np
import sys

from transformer import train, generate, load_and_generate, CharTokenizer


CONFIGS = {
    "small": {
        "d_model": 64,
        "n_heads": 4,
        "d_ff": 256,
        "n_layers": 2,
        "seq_len": 64,
        "batch_size": 8,
        "lr": 3e-3,
    },
    "medium": {
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_layers": 4,
        "seq_len": 128,
        "batch_size": 16,
        "lr": 1e-3,
    },
    "large": {
        "d_model": 256,
        "n_heads": 8,
        "d_ff": 1024,
        "n_layers": 6,
        "seq_len": 256,
        "batch_size": 8,
        "lr": 5e-4,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Train a character-level transformer LM.")
    parser.add_argument("--text", default="data/sample.txt", help="Path to training text file.")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps.")
    parser.add_argument("--config", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--checkpoint", default="checkpoint.pkl", help="Where to save the model.")
    parser.add_argument("--generate", action="store_true", help="Generate text after training.")
    parser.add_argument("--prompt", default="The ", help="Prompt for text generation.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.text):
        print(f"Text file not found: {args.text}")
        sys.exit(1)

    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text)} characters from {args.text}")

    config = CONFIGS[args.config].copy()
    config["seed"] = args.seed

    params, tokenizer, losses = train(
        text,
        config,
        n_steps=args.steps,
        log_every=args.log_every,
        checkpoint_path=args.checkpoint,
    )

    if args.generate:
        print("\n--- Generated text ---")
        out = generate(
            args.prompt, params, tokenizer, config,
            max_new=300, temperature=args.temperature, top_k=args.top_k
        )
        print(out)

    # Plot loss curve if matplotlib available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training loss")
        plt.tight_layout()
        plt.savefig("loss_curve.png")
        print("Loss curve saved to loss_curve.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
