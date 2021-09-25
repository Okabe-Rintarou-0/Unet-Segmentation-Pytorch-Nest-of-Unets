import torch
import torchvision
from Models import U_Net

pthUrl = r'D:/Unet-Segmentation-Pytorch-Nest-of-Unets/model/Unet_D_15_4/Unet_epoch_15_batchsize_4.pth'

ptUrl = r'D:/Unet-Segmentation-Pytorch-Nest-of-Unets/model/Unet_D_15_4/Unet_epoch_15_batchsize_4.pt'

model = U_Net(3, 1)
print(pthUrl)
test1 = model.load_state_dict(torch.load(pthUrl))
model.eval()
example = torch.rand(4, 1, 608, 608)
traced_module = torch.jit.trace(model, example)
traced_module.save(ptUrl)
