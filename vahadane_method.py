import staintools
import matplotlib.pyplot as plt
from IQA_pytorch import SSIM, utils
from PIL import Image
import torch

REINHARD = 'reinhard'
VAHADANE = 'vahadane'
MACENKO = 'macenko'

# Read data
target = staintools.read_image("C:\\Users\\yeeon\\Stain_normalization\\img_patch\\experiments_target\\patch_new_256_1M01_12.png")   #new philips
to_transform = staintools.read_image("C:\\Users\\yeeon\\Stain_normalization\\img_patch\\experiments_transform\\patch_old_256_1M01_53.png")


# Stain normalize (vahadane)
normalizer = staintools.StainNormalizer(method=VAHADANE)
normalizer.fit(target)
transformed = normalizer.transform(to_transform)


plt.imsave('C:\\Users\\yeeon\\Stain_normalization\\vaha.png', transformed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
ref_path  = 'patch_new_256_1M01_12.png'
dist_path = 'vaha.png' 
 
ref = utils.prepare_image(Image.open(ref_path).convert("RGB")).to(device)
dist = utils.prepare_image(Image.open(dist_path).convert("RGB")).to(device)
 
model = SSIM(channels=3)
 
score = model(dist, ref, as_loss=False)
print('vaha_','score: %.4f' % score.item())