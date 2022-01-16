import math
import cv2
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
        
        
class FrequencyLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(FrequencyLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, x, y):
        x_fft = torch.fft.rfft2(x, dim=(2,3))
        y_fft = torch.fft.rfft2(y, dim=(2,3))
        loss = self.criterion(x_fft, y_fft)
        return loss


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class BandingLoss(nn.Module):
    def __init__(self, threshold1=2/255., threshold2=12/255., use_cuda=False):
        super(BandingLoss, self).__init__()

        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))


        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))
        self.median_filter = MedianPool2d(kernel_size=3)

        if self.use_cuda:
            self.gaussian_filter_horizontal = self.gaussian_filter_horizontal.cuda()
            self.gaussian_filter_vertical = self.gaussian_filter_vertical.cuda()
            self.sobel_filter_horizontal = self.sobel_filter_horizontal.cuda()
            self.sobel_filter_vertical = self.sobel_filter_vertical.cuda()
            self.directional_filter = self.directional_filter.cuda()
            self.median_filter = self.median_filter.cuda()


    # def rgb_to_y(self, image: torch.Tensor) -> torch.Tensor:
    #     r"""Convert an RGB image to YCbCr.

    #     Args:
    #         image (torch.Tensor): RGB Image to be converted to YCbCr.

    #     Returns:
    #         torch.Tensor: YCbCr version of the image.
    #     """

    #     # if not torch.is_tensor(image):
    #     #     raise TypeError("Input type is not a torch.Tensor. Got {}".format(
    #     #         type(image)))

    #     # if len(image.shape) < 3 or image.shape[-3] != 3:
    #     #     raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
    #     #                     .format(image.shape))

    #     # r = image[..., 0, :, :]
    #     # g = image[..., 1, :, :]
    #     # b = image[..., 2, :, :]

    #     delta = .5
    #     # cb: torch.Tensor = (image[..., 2, :, :] - y) * .564 + delta
    #     # cr: torch.Tensor = (image[..., 0, :, :] - y) * .713 + delta
    #     # return torch.clip(torch.stack((y, cb, cr), -3), 0, 1.)
    #     return y


    def forward(self, img):
        # img_r = img[:,0:1]
        # img_g = img[:,1:2]
        # img_b = img[:,2:3]
        # print(img.shape)
        # img_y = self.rgb_to_y(img)
        img_y = .299 * img[..., 0:1, :, :] + .587 * img[..., 1:2, :, :] + .114 * img[..., 2:3, :, :]

        # img_y = img_yuv[:, 0:1, :, :]  ## Avoid inplace ops. Refer https://github.com/DCurro/CannyEdgePytorch/issues/5

        # blur_horizontal = self.gaussian_filter_horizontal(img_r)
        # blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        # blur_horizontal = self.gaussian_filter_horizontal(img_g)
        # blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        # blur_horizontal = self.gaussian_filter_horizontal(img_b)
        # blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_y)
        blurred_img = self.gaussian_filter_vertical(blur_horizontal)

        # blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        # blurred_img = torch.stack([torch.squeeze(blurred_img)])

        # grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        # grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        # grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        # grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        # grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        # grad_y_b = self.sobel_filter_vertical(blurred_img_b)
        grad_x = self.sobel_filter_horizontal(blurred_img)
        grad_y = self.sobel_filter_vertical(blurred_img)

        # COMPUTE THICK EDGES

        # grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        # grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        # grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        # grad_orientation = (torch.atan2(grad_y, grad_x) * (180.0/3.14159))
        # grad_orientation += 180.0
        # grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        # all_filtered = self.directional_filter(grad_mag)

        # inidices_positive = (grad_orientation / 45) % 8
        # inidices_negative = ((grad_orientation / 45) + 4) % 8

        # height = inidices_positive.size()[2]
        # width = inidices_positive.size()[3]
        # pixel_count = height * width
        # print(pixel_count)
        # pixel_range = torch.FloatTensor([range(pixel_count)])
        # if self.use_cuda:
        #     pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        # indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        # channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        # indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        # channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        # channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        # is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        # is_max = torch.unsqueeze(is_max, dim=0)

        # thin_edges = grad_mag.clone()
        # thin_edges[is_max==0] = 0.0

        # THRESHOLD

        # thresholded = thin_edges.clone()
        # thresholded[thin_edges<self.threshold1] = 0.0
        # thresholded[thin_edges>self.threshold2] = 0.0
        # thresholded[thin_edges<self.threshold1] = 0.0


        # Threshold on Grad_mag to get banding areas
        # flat_pixels = torch.zeros_like(grad_mag)
        # flat_pixels[grad_mag<self.threshold1] = 1
        # flat_pixels[grad_mag>=self.threshold1] = 0
        # flat_pixels = self.median_filter(flat_pixels)

        text_pixels = torch.zeros_like(grad_mag)
        # text_pixels[grad_mag>self.threshold2] = 0
        text_pixels[grad_mag<=self.threshold2] = 1
        text_pixels = text_pixels.int().float()
        if self.use_cuda:
            text_pixels = text_pixels.cuda()
        # text_pixels = self.median_filter(text_pixels)

        grad_mag = grad_mag * text_pixels

        # threshold thin edges
        # thresholded = thresholded * bband_mask.to(dtype=torch.float)

        # bband_scores = early_threshold.mean([2, 3], keepdim=False)
        # print(bband_score)
        # print(bband_score.shape)

        # assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        bband_loss = torch.mean(grad_mag)

        return bband_loss


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
        kernel_angle = kernel_angle * is_diag   # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight.data.copy_(torch.from_numpy(gaussian_2D))

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight.data.copy_(torch.from_numpy(sobel_2D))


        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight.data.copy_(torch.from_numpy(sobel_2D.T))


        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight.data.copy_(torch.from_numpy(directional_kernels))


        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight.data.copy_(torch.from_numpy(hysteresis))


    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        canny_loss = torch.mean(thin_edges)


        return canny_loss
