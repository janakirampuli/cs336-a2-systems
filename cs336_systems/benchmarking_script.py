import argparse
import timeit
import torch
import numpy as np

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.cross_entropy import cross_entropy

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

    # device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use")

    # mixed precision
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16"], help="compute precision")

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

    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    X = torch.randint(0, args.vocab_size, (args.batch_size, args.max_seq_len), device=device)
    Y = torch.randint(0, args.vocab_size, (args.batch_size, args.max_seq_len), device=device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def run_step():
        if args.backward:
            model.zero_grad(set_to_none=True)

        with autocast_context():
            logits = model(X)
            if args.backward:
                loss = cross_entropy(logits, Y)

        if args.backward:
            loss.backward()

        synchronize()
    
    if args.warmup_steps > 0:
        print(f"performing warmup for {args.warmup_steps} steps...")
        for _ in range(args.warmup_steps):
            run_step()
    
    print(f'benchmarking...')
    times = []

    for i in range(args.n_steps):
        start_t = timeit.default_timer()
        run_step()
        end_t = timeit.default_timer()
        times.append(end_t - start_t)

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    pass_type = "forward + backward" if args.backward else "forward only"
    
    print(f"results ({pass_type}):")
    print(f"mean time: {mean_time*1000:.2f} ms")
    print(f"std dev: {std_time*1000:.2f} ms")

if __name__ =="__main__":
    main()