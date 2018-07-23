import torch
from torchvision.models import resnet18

def main():
    assert torch.cuda.is_available()
    try:
        device = torch.device("cuda")
        model = resnet18().to(device)
        times = 1000
        for _ in range(times):
            print(model(torch.randn(32, 3, 224, 224).to(device)).sum())
    except KeyboardInterrupt:
        print("Interrupted. Releasing resources...")
    finally:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
