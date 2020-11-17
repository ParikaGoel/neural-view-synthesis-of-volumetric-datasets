import torch
from model import ssim
# from lpips_pytorch import LPIPS

# Ref: https://github.com/VainF/pytorch-msssim
# msssim_criterion = ssim.MS_SSIM(data_range=1.0, size_average=True, win_size=5, channel=3)

# Ref: https://github.com/S-aiueo32/lpips-pytorch
# lpips_criterion = LPIPS(net_type='alex',version='0.1')

class MS_SSIM_Loss(ssim.MS_SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )

class SSIM_Loss(ssim.SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )

ssim_criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=1)
msssim_criterion = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)

def prior_loss(output, validity_mask):
    loss = output.clone()
    loss = loss * validity_mask
    mask = torch.ge(loss, 0.0) & torch.le(loss, 1.0)
    loss[mask] = 0.0
    mask = torch.gt(loss, 1.0)
    loss[mask] = loss[mask] - 1.0
    mask = torch.lt(loss, 0.0)
    loss[mask] = torch.abs(loss[mask])

    n_valid_vals = torch.sum(validity_mask == True)
    loss = torch.sum(loss) / n_valid_vals
    return loss


def mse(target, output, validity_mask):
    loss = torch.nn.functional.mse_loss(target, output, reduction='none')
    loss = loss * validity_mask

    n_valid_vals = torch.sum(validity_mask == True)
    loss = torch.sum(loss) / n_valid_vals
    return loss


def l1(target, output, validity_mask):
    loss = torch.nn.functional.l1_loss(target, output, reduction='none')
    loss = loss * validity_mask

    n_valid_vals = torch.sum(validity_mask == True)
    loss = torch.sum(loss) / n_valid_vals
    return loss


def msssim(target, output, validity_mask):
    device = target.device
    msssim_criterion.to(device)

    output = torch.sigmoid(output)

    target = target * validity_mask 
    output = output * validity_mask

    return msssim_criterion(output, target)


def psnr(target, output, validity_mask):
    mse_val = mse(target, output, validity_mask)
    psnr_val = 10 * torch.log10(1.0 / mse_val)
    return psnr_val


# def lpips(target, output, validity_mask):
#     target = target * validity_mask
#     output = output * validity_mask

#     device = target.device
#     lpips_criterion.to(device)
#     lpips_val = lpips_criterion(output, target)
#     return lpips_val


def bce(target, output, validity_mask):
    tgt_fg_mask = torch.eq(target[:,0,...],0) & torch.eq(target[:,1,...],0) & torch.eq(target[:,2,...],0)
    tgt_fg_mask = ~tgt_fg_mask.unsqueeze_(1)
    output_fg_mask = torch.nn.Sigmoid()(output)
    mask = validity_mask[:,0,:,:].unsqueeze(1)
    tgt_fg_mask = tgt_fg_mask * mask
    output_fg_mask = output_fg_mask * mask
    loss = torch.nn.functional.binary_cross_entropy(output_fg_mask.float(), tgt_fg_mask.float(), reduction='none')
    loss = loss * mask
    loss = torch.sum(loss)
    n_valid_vals = torch.sum(mask == True)
    loss = loss / n_valid_vals
    return loss 


def loss_fft(target, output, validity_mask):
    target = target * validity_mask
    output = output * validity_mask

    target_fft = torch.rfft(target,signal_ndim=2, onesided=False)
    output_fft = torch.rfft(output,signal_ndim=2, onesided=False)

    loss_real = l1(target_fft[...,0], output_fft[...,0], validity_mask)
    loss_img = l1(target_fft[...,1], output_fft[...,1], validity_mask)

    loss = loss_real + loss_img

    return loss
