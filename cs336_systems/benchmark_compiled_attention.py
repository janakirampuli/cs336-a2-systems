import torch
from torch import einsum
from typing import Optional
import math

torch.set_float32_matmul_precision('high')

def softmax(x, dim):
    return torch.softmax(x, dim=dim)

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor]=None
) -> torch.Tensor:
    d_k = K.size(-1)
    # shape: (batch_size, n_heads, max_seq_len, max_seq_len)
    scores = einsum('... q d, ... k d -> ... q k', Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)

    probs = softmax(scores, -1)

    # shape: (batch_size, n_heads, max_seq_len, d_model)
    attn = einsum('... q k, ... k v -> ... q v', probs, V)
    return attn


def run_benchmark():

    device = torch.device('cuda')

    try:
        compiled_attention = torch.compile(scaled_dot_product_attention)
    except:
        print(f'torch.compile failed')
        compiled_attention = scaled_dot_product_attention

    BATCH_SIZE = 8
    NUM_HEADS = 1
    D_MODEL_VALS = [16, 32, 64, 128]
    SEQ_LEN_VALS = [256, 1024, 4096, 8192, 16384]

    results = []

    print(f"{'seq_len':<10} | {'d_model':<10} | {'fwd_time (ms)':<15} | {'cmp_fwd_time (ms)':<20} | {'fwd_speedup':<12} | {'bwd_time (ms)':<15} | {'cmp_bwd_time (ms)':<20} | {'bwd_speedup':<12} | {'mem (MB)':<15} | {'cmp_memory (MB)':<15} | {'status':<10}")


    for seq_len in SEQ_LEN_VALS:
        for d_model in D_MODEL_VALS:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                shape = (BATCH_SIZE, NUM_HEADS, seq_len, d_model)

                Q = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
                K = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
                V = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)

                # warmup
                for _ in range(5):
                    out = scaled_dot_product_attention(Q, K, V)
                    loss = out.sum()
                    loss.backward()
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                
                torch.cuda.synchronize()

                # forward(100)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(100):
                    _ = scaled_dot_product_attention(Q, K, V)
                end_event.record()
                torch.cuda.synchronize()

                fwd_time_ms = start_event.elapsed_time(end_event) / 100

                # measure memory before backward
                # torch.cuda.reset_peak_memory_stats()

                # out = scaled_dot_product_attention(Q, K, V)

                # mem_bytes = torch.cuda.max_memory_allocated()
                # mem_mb = mem_bytes / (1024*1024)

                grad = torch.rand_like(out)

                start_event.record()
                for _ in range(100):
                    temp_out = scaled_dot_product_attention(Q, K, V)
                    temp_out.backward(grad)
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                end_event.record()
                torch.cuda.synchronize()

                total_time_ms = start_event.elapsed_time(end_event) / 100
                bwd_time_ms = total_time_ms - fwd_time_ms

                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)


                # compiled computation

                # warmup
                
                for _ in range(5):
                    out = compiled_attention(Q, K, V)
                    loss = out.sum()
                    loss.backward()
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                # forward(100)

                start_event.record()
                for _ in range(100):
                    _ = compiled_attention(Q, K, V)
                end_event.record()
                torch.cuda.synchronize()

                cmp_fwd_time_ms = start_event.elapsed_time(end_event) / 100

                grad = torch.rand_like(out)

                start_event.record()
                for _ in range(100):
                    temp_out = compiled_attention(Q, K, V)
                    temp_out.backward(grad)
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                end_event.record()
                torch.cuda.synchronize()

                cmp_total_time_ms = start_event.elapsed_time(end_event) / 100
                cmp_bwd_time_ms = cmp_total_time_ms - cmp_fwd_time_ms

                cmp_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

                fwd_speedup = fwd_time_ms / cmp_fwd_time_ms if cmp_fwd_time_ms > 0 else 0
                bwd_speedup = bwd_time_ms / cmp_bwd_time_ms if cmp_bwd_time_ms > 0 else 0

                results.append({
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "fwd_time (ms)": fwd_time_ms,
                    "cmp_fwd_time (ms)": cmp_fwd_time_ms,
                    "fwd_speedup": fwd_speedup,
                    "bwd_time (ms)": bwd_time_ms,
                    "cmp_bwd_time (ms)": cmp_bwd_time_ms,
                    "bwd_speedup": bwd_speedup,
                    "memory (MB)": mem_mb,
                    "cmp_memory (MB)": cmp_mem_mb,
                    "status": "OK"
                })

                print(f"{seq_len:<10} | {d_model:<10} | {fwd_time_ms:<15.4f} | {cmp_fwd_time_ms:<20.4f} | {fwd_speedup:<12.4f} | {bwd_time_ms:<15.4f} | {cmp_bwd_time_ms:<20.4f} | {bwd_speedup:<12.4f} | {mem_mb:<15.2f} | {cmp_mem_mb:<15.2f} | {'OK':<10}")

                del Q, K, V, out, grad

            except torch.cuda.OutOfMemoryError:
                results.append({
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "fwd_time (ms)": None,
                    "cmp_fwd_time (ms)": None,
                    "fwd_speedup": None,
                    "bwd_time (ms)": None,
                    "cmp_bwd_time (ms)": None,
                    "bwd_speedup": None,
                    "memory (MB)": None,
                    "cmp_memory (MB)": None,
                    "status": "OOM"
                })
                print(f"{seq_len:<10} | {d_model:<10} | {'N/A':<15} | {'N/A':<20} | {'N/A':<12} | {'N/A':<15} | {'N/A':<20} | {'N/A':<12} | {'OOM':<15} | {'OOM':<15} | {'FAIL':<10}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"error for seq_len:{seq_len}, d_model:{d_model}: {e}")


if __name__ == "__main__":
    run_benchmark()



