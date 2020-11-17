#%%
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim

from PIL import Image

img = Image.open('./img1.png')
img.show()
img = np.array(img)[...,:3].astype(np.float) / 255.

print(img)

#%%
# img = img_as_float(data.camera())
rows, cols, _ = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1

print(noise.shape)

#%%
def mse(x, y):
    return np.linalg.norm(x - y)

#%%
img_noise = img + noise
img_const = img + abs(noise)

mse_none = mse(img, img)
ssim_none, grad_none = ssim(img, img, data_range=img.max() - img.min(),
                    gradient=True,
                    multichannel=True, gaussian_weights=True,
                    sigma=1.5, use_sample_covariance=False)

mse_noise = mse(img, img_noise)
ssim_noise, grad_noise = ssim(img, img_noise, gradient=True,
                  data_range=img_noise.max() - img_noise.min(), 
                  multichannel=True, gaussian_weights=True,
                  sigma=1.5, use_sample_covariance=False)

mse_const = mse(img, img_const)
ssim_const, grad_const = ssim(img, img_const, gradient=True,
                  data_range=img_const.max() - img_const.min(), 
                  multichannel=True, gaussian_weights=True,
                  sigma=1.5, use_sample_covariance=False)

label = 'MSE: {:.2f}, SSIM: {:.2f}'

#%%

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img)
ax[0].set_xlabel(label.format(mse_none, ssim_none))
ax[0].set_title('Original image')

ax[1].imshow(img_noise)
ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
ax[1].set_title('Image with noise')

ax[2].imshow(img_const)
ax[2].set_xlabel(label.format(mse_const, ssim_const))
ax[2].set_title('Image plus constant')

plt.tight_layout()
plt.show()

# %%

