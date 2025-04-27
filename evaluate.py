from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm
import  cv2
import torch
from torch.utils.data import DataLoader
# from lut_read import *
from networks import Equi_convnext_tea,Equi_convnext_tea_omni
import datasets
from metrics import Evaluator
from saver import Saver
import numpy as np
parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Test")

parser.add_argument("--root", default="G:\\stanford2d3d\\", type=str, help="path to the dataset.")
parser.add_argument("--dataset", default="stanford2d3d", choices=["3d60", "panosuncg", "stanford2d3d", "matterport3d"],
                    type=str, help="dataset to evaluate on.")
parser.add_argument("--height", type=int, default=512, help="batch size")
parser.add_argument("--width", type=int, default=1024, help="batch size")
parser.add_argument("--load_weights_dir",default='G:\liujingguo\\five\YUHAN', type=str, help="folder of model to load")

parser.add_argument("--num_workers", type=int, default=0, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_classes", type=int, default=13, help="number of dataloader workers")

parser.add_argument("--median_align", action="store_true", help="if set, apply median alignment in evaluation")
parser.add_argument("--save_samples", default=True, help="if set, save the depth maps and point clouds")

settings = parser.parse_args()

def change_feat(img):
    img = torch.mean(img, dim=1).squeeze().cpu().numpy()
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_weights_folder = os.path.expanduser(settings.load_weights_dir)
    model_path = os.path.join(load_weights_folder, "model_iou_best_1.pth")
    model_dict = torch.load(model_path)
    print(model_dict['miou'])
    # data
    datasets_dict = {"stanford2d3d": datasets.S2d3dSemgDataset,
                     "matterport3d": datasets.Matterport3D,
                     "struct3d": datasets.Struct3D,
                     "OmniMNIST": datasets.OmniMNIST}  ##s2d3d segmentation             mattort3d and 3d60 depth estimation
    dataset = datasets_dict[settings.dataset]
    # ################################STANFORD2D3D
    test_dataset = dataset(settings.root, depth=False,
                                    hw=(settings.height, settings.width), fold='1_valid')
    test_loader = DataLoader(test_dataset, settings.batch_size, False,
                                 num_workers=settings.num_workers, pin_memory=True, drop_last=False)
    invalid_ids = []
    label_weight = torch.load('G:/liujingguo/segmentation/networks/label13_weight.pth').float().to(device)
    label_weight[invalid_ids] = 0
    label_weight *= (settings.num_classes - len(invalid_ids)) / label_weight.sum()
    colors = np.load('G:/Stanford2D3D_sem/colors.npy')
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // settings.batch_size
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    Net_dict = {"convequi": Equi_convnext_tea}  # Equi_convnext_tea_omni
    Net_dep = Net_dict["convequi"]

    model = Net_dep(settings.height, settings.width)

    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    evaluator = Evaluator(settings.median_align)
    evaluator.reset_eval_metrics()
    saver = Saver(load_weights_folder)
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")
    vis_dir = ''#'G:\liujingguo\\five\\vis 512 1\\'  # G:\liujingguo\\fuse\\vis\\
    cm = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):
            # if batch_idx == 11:break
            for key, ipt in inputs.items():
                if torch.is_tensor(ipt):
                    inputs[key] = ipt.to(device)
            rgb = inputs["x"]
            gt_sem = inputs["sem"]
            mask = (gt_sem >= 0).to(device)
            if mask.sum() == 0:
                continue
            B, C, H, W = rgb.shape
            outputs = model(rgb)
            pred_sem = outputs["sem"].detach()

            gt = gt_sem[mask]
            pred = pred_sem.argmax(1)[mask]
            if batch_idx % 1 == 0:
                if vis_dir:
                    import matplotlib.pyplot as plt
                    from imageio import imwrite
                    cmap = (plt.get_cmap('gist_rainbow')(np.arange(settings.num_classes) / settings.num_classes)[
                            ..., :3] * 255).astype(np.uint8)
                    rgb = (inputs['x'][0, :3].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
                    imwrite(os.path.join(vis_dir, inputs['fname'][0].strip() + '.rgb.png'), rgb)
                    pre0 = pred_sem[0]
                    vis_sem = cmap[pre0.argmax(0).cpu().numpy()]
                    vis_sem = (vis_sem * 1.0).astype(np.uint8)#rgb * 0.2 +
                    imwrite(os.path.join(vis_dir, inputs['fname'][0].strip()), vis_sem)
                    mid = gt_sem[0]
                    vis_sem = cmap[mid.cpu().numpy()]
                    vis_sem = (vis_sem * 1.0).astype(np.uint8)#rgb * 0.2 +
                    imwrite(os.path.join(vis_dir, inputs['fname'][0].strip() + '.gt.png'), vis_sem)
            assert gt.min() >= 0 and gt.max() < settings.num_classes and pred_sem.shape[1] == settings.num_classes
            cm += np.bincount((gt * settings.num_classes + pred).cpu().numpy(),
                              minlength=settings.num_classes ** 2)
        print('  Summarize  '.center(50, '='))
        cm = cm.reshape(settings.num_classes, settings.num_classes)
        id2class = np.array(test_dataset.ID2CLASS)
        valid_mask = (cm.sum(1) != 0)
        cm = cm[valid_mask][:, valid_mask]
        id2class = id2class[valid_mask]
        inter = np.diag(cm)
        union = cm.sum(0) + cm.sum(1) - inter
        ious = inter / union
        accs = inter / cm.sum(1)
        for name, iou, acc in zip(id2class, ious, accs):
            print(f'{name:20s}:    iou {iou * 100:5.2f}    /    acc {acc * 100:5.2f}')
        print(f'{"Overall":20s}:    iou {ious.mean() * 100:5.2f}    /    acc {accs.mean() * 100:5.2f}')


if __name__ == "__main__":
    main()