
import torch

print(torch.version.cuda)
print('Cuda: ', torch.cuda.is_available())
print('Cuda: ', torch.cuda.current_device())
print(torch.cuda.get_device_name())
cihazim = torch.cuda.current_device()
print(torch.cuda.get_device_capability(cihazim))

# To use your GPU during these processes
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x,y = x.to(device), y.to(device)
# model = SimpleANN().to(device)


















