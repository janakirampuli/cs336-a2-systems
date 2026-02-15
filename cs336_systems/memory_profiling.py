import argparse
import timeit
import torch
import numpy as np

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import AdamW

from contextlib import nullcontext

def parse_args():
    parser = argparse.ArgumentParser()

    # model hyperparams
    parser.add_argument("--d_model", type=int, default=512, help="embedding dimension (d_model)")
    parser.add_argument("--n_layers", type=int, default=4, help="number of transformer layers (num_layers)")
    parser.add_argument("--n_heads", type=int, default=16, help="number of attention heads (num_heads)")
    parser.add_argument("--d_ff", type=int, default=1344, help="feed forward dimension")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length (context_length)")

    # benchmarking params
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--warmup_steps", type=int, default=5, help="number of warmup steps")
    parser.add_argument("--n_steps", type=int, default=10, help="number of steps to measure")
    parser.add_argument("--backward", action="store_true", help="if set, benchmark both forward, backward pass")
    parser.add_argument("--include_optimizer", action="store_true", help="include optimizer step")

    # device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use")

    # mixed precision
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16"], help="compute precision")

    # memory profiling
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "full_step"])

    parser.add_argument("--profile", action="store_true", help="pytorch mem snapshot")
    parser.add_argument("--output_file", type=str, default="memory_snapshot.pickle")

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"running on device {device}")

    use_amp = args.precision == "bf16" and device.type == "cuda"

    if use_amp:
        print("using bf16 mixed precision")
        autocast_context = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        print("using fp32")
        autocast_context = nullcontext

    d_ff = args.d_ff if args.d_ff is not None else 4 * args.d_model

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.max_seq_len,
        "num_layers": args.n_layers,
        "d_model": args.d_model,
        "num_heads": args.n_heads,
        "d_ff": d_ff,
        "device": device
    }

    print(f"tarnsformerLM config: {model_config}")

    model = TransformerLM(**model_config)
    model.to(device)
    model.train()

    optimizer = None

    if args.include_optimizer:
        optimizer = AdamW(model.parameters(), lr=3e-3)
        args.backward = True

    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    X = torch.randint(0, args.vocab_size, (args.batch_size, args.max_seq_len), device=device)
    Y = torch.randint(0, args.vocab_size, (args.batch_size, args.max_seq_len), device=device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def run_step():
        if args.backward or args.include_optimizer:
            if optimizer:
                optimizer.zero_grad()
            else:
                model.zero_grad(set_to_none=True)

        logits = model(X)
        if args.backward or args.include_optimizer:
            loss = cross_entropy(logits, Y)

        if args.backward or args.include_optimizer:
            loss.backward()

        if args.include_optimizer:
            optimizer.step()

        synchronize()
    
    if args.warmup_steps > 0:
        print(f"performing warmup for {args.warmup_steps} steps...")
        for _ in range(args.warmup_steps):
            run_step()
    
    print(f'profiling memory...')

    if args.profile and device.type == "cuda":
        print(f'recording memory to {args.output_file}')
        torch.cuda.memory._record_memory_history(max_entries=1000000)

        for i in range(args.n_steps):
            run_step()
            
        torch.cuda.memory._dump_snapshot(args.output_file)
        torch.cuda.memory._record_memory_history(enabled=None)

        print("snapshot saved")
    

if __name__ =="__main__":
    main()