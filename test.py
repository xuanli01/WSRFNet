#!/usr/bin/python3
#coding=utf-8
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import dataset
from model import WSRFNet
from eval_metrics import *

class Test(object):
    def __init__(self, args):
        ## dataset
        self.eval_dataset = args.dataset
        self.vis_results = args.vis_results
        self.cfg = dataset.Config(eval_dataset = args.dataset,
                                  datapath = args.data_root, 
                                  modelpath = args.checkpoint)
        self.data = dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.CLASSES = ['bg', 'EX', 'HE', 'SE', 'MA'] # bg:background
        self.net = WSRFNet(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.palette = np.array([
            [0, 0, 0],     # bg: black
            [255, 0, 0],   # EX: red
            [0, 255, 0],   # HE: green
            [255, 255, 0], # SE: yellow
            [0, 128, 255]  # MA: blue
        ])
    

    def show_result(self, result, name):
        vis_dir = self.eval_dataset + '_vis'
        os.makedirs(vis_dir, exist_ok=True)

        seg = result.copy()
        seg[seg<=0.5]=0
        bg = np.zeros((seg.shape[1], seg.shape[2]), dtype=np.uint8)
        bg[np.sum(seg,axis=0)==0]=2
        bg = bg[np.newaxis,:]
        seg = np.concatenate((bg,seg),axis=0)
        seg = np.argmax(seg,axis=0)

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(self.palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]
        color_seg = color_seg.astype(np.uint8)

        cv2.imwrite(os.path.join(vis_dir,name[0]+'.png'),color_seg)

    def predict(self):
        print("start predict....")
        pred_list = []
        mask_list = []
        with torch.no_grad():
            for image, mask, shape, name in tqdm(iter(self.loader)):

                image = image.cuda().float()
                out_Pf = self.net(image, shape)
                pred = out_Pf.cpu().numpy()
                pred_list.append(pred)
                mask_list.append(mask[0].cpu().numpy())

                if self.vis_results:
                    self.show_result(pred, name)
                    
            # Evaluation
            num_classes = len(self.CLASSES)
            aupr, dice, iou = metrics(
                pred_list, mask_list, num_classes) 
            eval_results = {}    
            summary_str = ''
            summary_str += 'per class results:\n'

            line_format = '{:<15} {:>10} {:>10} {:>10}\n'
            summary_str += line_format.format('Class', 'AUPR', 'Dice', 'IoU')
            if self.CLASSES is None:
                class_names = tuple(range(num_classes))
            else:
                class_names = self.CLASSES
            for i in range(1, num_classes):
                dice_str = '{:.2f}'.format(dice[i] * 100)
                iou_str = '{:.2f}'.format(iou[i] * 100)
                aupr_str = '{:.2f}'.format(aupr[i] * 100)
                summary_str += line_format.format(class_names[i], aupr_str, dice_str, iou_str)

            mIoU = np.nanmean(np.nan_to_num(iou[-4:], nan=0))
            mdice = np.nanmean(np.nan_to_num(dice[-4:], nan=0))
            mAUPR = np.nanmean(np.nan_to_num(aupr[-4:], nan=0))

            summary_str += 'Summary:\n'
            line_format = '{:<15} {:>10} {:>10} {:>10}\n'
            summary_str += line_format.format('Scope', 'mAUPR', 'mDice', 'mIoU')

            iou_str = '{:.2f}'.format(mIoU * 100)
            dice_str = '{:.2f}'.format(mdice * 100)
            aupr_str = '{:.2f}'.format(mAUPR * 100)
            summary_str += line_format.format('Global', aupr_str, dice_str, iou_str)

            eval_results['mIoU'] = mIoU
            eval_results['mDice'] = mdice
            eval_results['mAUPR'] = mAUPR

            print(summary_str)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='WSRFNet')
    parser.add_argument('-d', '--dataset', help='enter a dataset to evaluate')
    parser.add_argument('-p', '--data_root', help='the root of test set')
    parser.add_argument('-c', '--checkpoint', help='the path of the checkpoint')
    parser.add_argument('--vis_results', action='store_true', help='Visualize the segmentation maps or not')
    args = parser.parse_args()
    test = Test(args)
    test.predict()

