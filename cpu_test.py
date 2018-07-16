import torch
from torchvision.models import resnet18

def main():
    assert torch.cuda.is_available()
    device = torch.device("cpu")
    model = resnet18().to(device)
    times = 1000
    for _ in range(times):
        print(model(torch.randn(32, 3, 224, 224).to(device)).sum())

if __name__ == '__main__':
    main()
