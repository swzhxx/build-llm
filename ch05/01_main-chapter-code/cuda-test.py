import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
    # 创建一个张量并移动到 GPU 上
    x = torch.tensor([1.0, 2.0]).to(device)
    print(x)
else:
    print("CUDA is not available.")