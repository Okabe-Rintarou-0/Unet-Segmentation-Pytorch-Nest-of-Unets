import torch
import torchvision
from PIL import Image

from Models import U_Net

data_transform = torchvision.transforms.Compose([
    #  torchvision.transforms.Resize((128,128)),
    #   torchvision.transforms.CenterCrop(96),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model_path = r'D:\Unet-Segmentation-Pytorch-Nest-of-Unets\model\Unet_D_15_4\Unet_epoch_15_batchsize_4.pth'
test_image = './dataset_for_test/train/image/00001.png'

model = U_Net(3, 1)
model.load_state_dict(torch.load(model_path))
model.eval()

im_tb = Image.open(test_image)
s_tb = data_transform(im_tb)

pred_tb = model(s_tb.unsqueeze(0).to("cpu")).cpu()
pred_tb = torch.sigmoid(pred_tb)
print(pred_tb)
pred_tb = pred_tb.detach().numpy()[0][0]
print(pred_tb)
print(pred_tb.shape())
