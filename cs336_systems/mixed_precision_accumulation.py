import torch

print("1. FP32 Accumulation")
# Baseline: Standard 32-bit float accumulation
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(f"Result: {s.item():.8f}")

print("\n2. FP16 Accumulation")
# Pure FP16: Both the accumulator and the update value are FP16
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"Result: {s.item():.8f}")

print("\n3. Mixed Precision (FP32 Accumulator + FP16 Value)")
# The tensor is created in FP16, but added to an FP32 accumulator.
# PyTorch promotes the operation to FP32.
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"Result: {s.item():.8f}")

print("\n4. Mixed Precision (Explicit Cast)")
# Explicitly casting the FP16 value to FP32 before adding.
# Ideally identical to case 3.
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(f"Result: {s.item():.8f}")