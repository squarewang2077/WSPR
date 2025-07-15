from robustbench.utils import load_model
# import torch
 
 
# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("Number of GPUs:", torch.cuda.device_count())
# print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

model = load_model('Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
print(model.__class__)
