import torch
import torchvision
from Models import U_Net

pthUrl = r'D:/Unet-Segmentation-Pytorch-Nest-of-Unets/model/Unet_D_15_4/Unet_epoch_15_batchsize_4.pth'

ptUrl = r'D:/Unet-Segmentation-Pytorch-Nest-of-Unets/model/Unet_D_15_4/model.ptl'

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = U_Net(3, 1)
model.load_state_dict(torch.load(pthUrl))
model.eval()
example = torch.rand(1, 3, 608, 608)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter(ptUrl)
