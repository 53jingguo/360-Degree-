
import os
import cv2
import numpy as np
from scipy import ndimage
import scipy
import scipy.io as sio
import torch
def biliner(input,cooordinate):
    height, width, C = input.shape
    x1, y1 = torch.floor(torch.from_numpy(np.array(cooordinate[0]))).type(torch.int), torch.floor(torch.from_numpy(np.array(cooordinate[1]))).type(torch.int)
    x2, y2 = torch.as_tensor(x1+1,dtype=torch.int),torch.as_tensor(y1+1,dtype=torch.int)
    x1, x2 = torch.clamp(x1, 0, width - 1).type(torch.long),torch.clamp(x2, 0, width - 1).type(torch.long)
    y1, y2 = torch.clamp(y1, 0, height - 1).type(torch.long), torch.clamp(y2, 0, height - 1).type(torch.long)
    dx1, dx2 = torch.from_numpy(np.array(cooordinate[0])).type(torch.float) - x1, x2 - torch.from_numpy(np.array(cooordinate[0])).type(torch.float)
    dy1, dy2 = torch.from_numpy(np.array(cooordinate[1])).type(torch.float) - y1, y2 - torch.from_numpy(np.array(cooordinate[1])).type(torch.float)
    dx1, dx2 = dx1.unsqueeze(1).repeat(1,C),dx2.unsqueeze(1).repeat(1,C)
    dy1, dy2 = dy1.unsqueeze(1).repeat(1,C),dy2.unsqueeze(1).repeat(1,C)
    interpolated_value = (input[y1, x1] * dx2 * dy2 +
                          input[y1, x2] * dx1 * dy2 +
                          input[y2, x1] * dx2 * dy1 +
                          input[y2, x2] * dx1 * dy1)
    return interpolated_value


im = r"G:\LUT\test0.png"
mask = cv2.imread(im)
mask_ori = cv2.resize(mask.astype("float"),dsize=(1024,128),interpolation=cv2.INTER_LINEAR)
cv2.imwrite(r"G:\LUT\bi.png",mask_ori)
mask = torch.as_tensor(mask_ori)
mask =mask.repeat(1,1,3)
LUT_name = r'G:\liujingguo\\five\LUT\LUT512\LUT_4x4.mat'
arr_LUT = sio.loadmat(LUT_name)['LUT']
tmp =np.array([arr_LUT[:,0], arr_LUT[:,1]])
mask_co = biliner(mask, [arr_LUT[:,0], arr_LUT[:,1]])
masks_co= mask_co.reshape(128*7,256*7,9)
masks_co = masks_co.numpy()
# cv2.imwrite(r"G:\LUT\bi_out.png", masks_co)



