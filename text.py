from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from model import resnet101
from torchvision import datasets, transforms
import cv2
from torch.nn import functional as F

from PIL import Image

Img = '0003'

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            #print(name)
            if name is "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []

    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

img_pil = Image.open(Img + '.jpg')

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

myresnet = resnet101()
myresnet.eval()

myresnet.load_state_dict(torch.load('Emotion6.pth'))
exact_list = ["layer4", 'fc']

params = list(myresnet.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

myexactor = FeatureExtractor(myresnet, exact_list)

x = myexactor(img_variable)

classes = {0: 'anger',
           1: 'disgust',
           2: 'fear',
           3: 'joy',
           4: 'sadness',
           5: 'surprise'}

h_x = F.softmax(x[1]).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

for i in range(0, 6):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

CAMs = returnCAM(x[0].detach().numpy(), weight_softmax, [idx[0]])

print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread(Img + '.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite(Img + 'Resnet101.jpg', result)
