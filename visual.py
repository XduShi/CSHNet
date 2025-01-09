from Evison import Display, show_network
import torch


model = torch.load('/media/yangxilab/DiskA/wzh/Transformer/Trans-p2phd/checkpoints/SA2Res4/90_net_G.pth')
print(model)

show_network(model)