import argparse
import timeit
import torch
import numpy as np
import torch.cuda.nvtx as nvtx
from torch import einsum
from typing import Optional
import math
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
import cs336_basics
from cs336_basics.cross_entropy import cross_entropy
import sys


from cs336_basics.adamw import AdamW
from cs336_basics.softmax import softmax

def annotated_scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor]=None
) -> torch.Tensor:

    with nvtx.range("scaled_dot_product_attention"):
        d_k = K.size(-1)

        with nvtx.range("computing attention scores"):
            scores = einsum('... q d, ... k d -> ... q k', Q, K) / math.sqrt(d_k)

        if mask is not None:
            with nvtx.range("masking"):
                scores = scores.masked_fill(~mask, -torch.inf)

        with nvtx.range("computing softmax"):
            probs = softmax(scores, -1)

        with nvtx.range("final matmul"):
            attn = einsum('... q k, ... k v -> ... q v', probs, V)

    return attn



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

    return parser.parse_args()

def main():
    args = parse_args()

    # cs336_basics.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    for name, module in list(sys.modules.items()):
        if name.startswith("torch") or name.startswith("numpy"):
            continue
            
        if hasattr(module, 'scaled_dot_product_attention'):
            try:
                setattr(module, 'scaled_dot_product_attention', annotated_scaled_dot_product_attention)
                print(f" -> Patched: {name}.scaled_dot_product_attention")
                patched_count += 1
            except Exception:
                pass

    device = torch.device(args.device)
    print(f"running on device {device}")

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

    def run_step(step_name="step"):
        with nvtx.range(step_name):
            # zero grad
            if args.backward or args.include_optimizer:
                if optimizer:
                    optimizer.zero_grad()
                else:
                    model.zero_grad(set_to_none=True)
            # forward
            with nvtx.range("forward_pass"):   
                logits = model(X)
                if args.backward or args.include_optimizer:
                    loss = cross_entropy(logits, Y)
            # backward 
            if args.backward or args.include_optimizer:
                with nvtx.range("backward_pass"):
                    loss.backward()
            
            # optimizer
            if args.include_optimizer:
                with nvtx.range("optimizer_step"):
                    optimizer.step()

            synchronize()
    
    if args.warmup_steps > 0:
        print(f"performing warmup for {args.warmup_steps} steps...")
        with nvtx.range("warmup_phase"):
            for i in range(args.warmup_steps):
                run_step(step_name=f"warmup_step_{i}")
    
    print(f'benchmarking...')
    times = []

    with nvtx.range("benchmark_phase"):
        for i in range(args.n_steps):
            start_t = timeit.default_timer()
            run_step(step_name=f"measure_step_{i}")
            end_t = timeit.default_timer()
            times.append(end_t - start_t)

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    pass_type = "forward only"

    if args.include_optimizer: pass_type = "forward + backward + optimizer"
    elif args.backward: pass_type = "foward + backward"
    
    print(f"results ({pass_type}):")
    print(f"mean time: {mean_time*1000:.2f} ms")
    print(f"std dev: {std_time*1000:.2f} ms")

if __name__ =="__main__":
    main()