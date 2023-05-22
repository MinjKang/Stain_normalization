import staintools
import matplotlib.pyplot as plt
from IQA_pytorch import SSIM, utils
from PIL import Image
import torch


# REINHARD = 'reinhard'
# VAHADANE = 'vahadane'
# MACENKO = 'macenko'

# # Read data
# target = staintools.read_image("C:\\Users\\yeeon\\Stain_normalization\\img_patch\\experiments_target\\patch_new_256_1M01_12.png")
# to_transform = staintools.read_image("C:\\Users\\yeeon\\Stain_normalization\\img_patch\\experiments_transform\\patch_old_256_1M01_53.png")


# # Stain normalize (reinhard)
# normalizer = staintools.ReinhardColorNormalizer()
# normalizer.fit(target)
# transformed = normalizer.transform(to_transform)


# plt.imsave('C:\\Users\\yeeon\\Stain_normalization\\reinhard_result.png', transformed)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
ref_path  = 'patch_new_256_1M01_12.png'
dist_path = 'reinhard_result.png' 
 
ref = utils.prepare_image(Image.open(ref_path).convert("RGB")).to(device)
dist = utils.prepare_image(Image.open(dist_path).convert("RGB")).to(device)
 
model = SSIM(channels=3)
 
score = model(dist, ref, as_loss=False)
print('reinhard_','score: %.4f' % score.item())
