import torch

def check_cuda_version():
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        print(f"CUDA Version: {cuda_version}")
        print(f"cuDNN Version: {cudnn_version}")
    else:
        print("CUDA is not available.")

def check_pytorch_version():
    pytorch_version = torch.__version__
    print(f"PyTorch Version: {pytorch_version}")

if __name__ == "__main__":
    check_cuda_version()
    check_pytorch_version()
    print("CUDA and PyTorch versions checked.")