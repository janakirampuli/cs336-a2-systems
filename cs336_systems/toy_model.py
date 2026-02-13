import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("input:", x.dtype)

        x = self.fc1(x)
        print("after fc1:", x.dtype)

        x = self.relu(x)
        print("after relu:", x.dtype)

        x = self.ln(x)
        print("after layernorm:", x.dtype)

        x = self.fc2(x)
        print("after fc2 (logits):", x.dtype)

        return x

def main():
    device = "cuda"
    model = ToyModel(32, 5).to(device)

    print("parameter dtype:", model.fc1.weight.dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(4, 32, device=device)
    target = torch.randint(0, 5, (4,), device=device)

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        print("logits outside model:", logits.dtype)

        loss = criterion(logits, target)
        print("loss dtype:", loss.dtype)


    scaler.scale(loss).backward()

    print("\nGRADIENT DTYPES:")
    for name, param in model.named_parameters():
        print(name, "grad dtype:", param.grad.dtype)

if __name__ == "__main__":
    main()
