import torch as th

device='cuda' if th.cuda.is_available() else 'cpu'
print(f"Using device: {device}")