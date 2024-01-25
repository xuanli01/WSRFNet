#!/usr/bin/python3
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelets import DWT_2D
from efficientnet_pytorch import EfficientNet

# WFPM+SRM
class RefineModule(nn.Module):
    def __init__(self, dim=64):
        super(RefineModule, self).__init__()
        self.dwt0 = DWT_2D(wave='haar')
        self.dwt1 = DWT_2D(wave='haar')
        self.dwt2 = DWT_2D(wave='haar')
        self.dwt3 = DWT_2D(wave='haar')

        self.sqch2 = nn.Sequential(
            nn.Conv2d(4*dim, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.sqch3 = nn.Sequential(
            nn.Conv2d(4*dim, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.sqch4 = nn.Sequential(
            nn.Conv2d(4*dim, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.sq0 = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.sq1 = nn.Sequential(
            nn.Conv2d(dim*5, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.sq2 = nn.Sequential(
            nn.Conv2d(dim*5, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.sq3 = nn.Sequential(
            nn.Conv2d(dim*5, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.sq4 = nn.Sequential(
            nn.Conv2d(dim*5, dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
      
        self.filter0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.filter1 = nn.Sequential(
            nn.Conv2d(4*dim, 4*dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(4*dim, 4*dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),
        )
        self.filter3 = nn.Sequential(
            nn.Conv2d(4*dim, 4*dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),
        )  
        self.filter4 = nn.Sequential(
            nn.Conv2d(4*dim, 4*dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),
        )  

        self.refine0 = conv_bn_relu(dim, dim, kernel=3, stride=1, padding=1)
        self.refine1 = conv_bn_relu(dim, dim, kernel=3, stride=1, padding=1)  
        self.refine2 = conv_bn_relu(dim, dim, kernel=3, stride=1, padding=1)
        self.refine3 = conv_bn_relu(dim, dim, kernel=3, stride=1, padding=1)
        self.refine4 = conv_bn_relu(dim, dim, kernel=3, stride=1, padding=1)

    def forward(self, ou_S_pre, in_B1, in_B2, in_B3, in_B4, in_B5):

        # Wavelet-Based Feedback Pyramid Module (WFPM)
        # level 1
        F_n1 = self.filter0(ou_S_pre) 

        # level 2
        F_n2 = self.dwt0(F_n1)
        F_n2 = self.filter1(F_n2) 

        # level 3
        F_n3 = self.dwt1(self.sqch2(F_n2))
        F_n3 = self.filter2(F_n3) 

        # level 4
        F_n4 = self.dwt2(self.sqch3(F_n3))
        F_n4 = self.filter3(F_n4) 

        # level 5
        F_n5 = self.dwt3(self.sqch4(F_n4))
        F_n5 = self.filter4(F_n5) 

        # Scale-Specific Refinement Module (SRM)
        # SRM1
        X_n1 = self.sq0(torch.cat([in_B1, F_n1], dim=1))   
        B_n1 = self.refine0(X_n1 + in_B1)

        # SRM2
        X_n2 = self.sq1(torch.cat([in_B2, F_n2], dim=1))
        B_n2 = self.refine1(X_n2 + in_B2)

        # SRM3
        X_n3 = self.sq2(torch.cat([in_B3, F_n3], dim=1))
        B_n3 = self.refine2(X_n3 + in_B3)

        # SRM4
        X_n4 = self.sq3(torch.cat([in_B4, F_n4], dim=1))
        B_n4 = self.refine3(X_n4 + in_B4)

        # SRM5
        X_n5 = self.sq4(torch.cat([in_B5, F_n5], dim=1))
        B_n5 = self.refine4(X_n5 + in_B5)

        return B_n1, B_n2, B_n3, B_n4, B_n5

class MFA(nn.Module):
    def __init__(self):
        super(MFA, self).__init__()
        self.cbr0 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.cbr1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.cbr2 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.cbr3 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.cbr4 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)

    def forward(self,in_B_n1, in_B_n2, in_B_n3, in_B_n4, in_B_n5):
        out_Dn5 = self.cbr4(in_B_n5)
        out_Dn4 = self.cbr3(in_B_n4 + upsample_like(out_Dn5, in_B_n4))
        out_Dn3 = self.cbr2(in_B_n3 + upsample_like(out_Dn4, in_B_n3))
        out_Dn2 = self.cbr1(in_B_n2 + upsample_like(out_Dn3, in_B_n2))
        out_Dn1 = self.cbr0(in_B_n1 + upsample_like(out_Dn2, in_B_n1))
        return out_Dn1

class WSRFNet(nn.Module):
    def __init__(self, cfg):
        super(WSRFNet, self).__init__()
        self.cfg = cfg
        self.loop = 4
        self.bkbone = EfficientNet.from_name('efficientnet-b1', include_top=False)
        
        self.tran0 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran2 = nn.Sequential(nn.Conv2d(40, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran3 = nn.Sequential(nn.Conv2d(112, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran4 = nn.Sequential(nn.Conv2d(1280, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))

        self.decoder1 = MFA()
        self.midlayer = RefineModule()

        self.spv1 = nn.Conv2d(64, 4, 3, 1, 1)

        self.load_state_dict(torch.load(self.cfg.modelpath, map_location='cuda:0'))      
        
    def forward(self, image, shape=None):

        # ---------- Encoder ----------
        endpoints = self.bkbone.extract_endpoints(image)
        bkbone_f0 = endpoints['reduction_1']
        bkbone_f1 = endpoints['reduction_2']
        bkbone_f2 = endpoints['reduction_3']
        bkbone_f3 = endpoints['reduction_4']
        bkbone_f4 = endpoints['reduction_6']

        tran_B1 = self.tran0(bkbone_f0)
        tran_B2 = self.tran1(bkbone_f1)
        tran_B3 = self.tran2(bkbone_f2)
        tran_B4 = self.tran3(bkbone_f3)
        tran_B5 = self.tran4(bkbone_f4)

        # ---------- Decoder ----------

        # Recurrent Refinement
        # loop number n=1
        out_S = self.decoder1(tran_B1, tran_B2, tran_B3, tran_B4, tran_B5)

        out_S_list = []
        out_S_list.append(out_S)

        # loop number n>1
        for n in range(1, self.loop):
            B_n = self.midlayer(out_S_list[n-1], tran_B1, tran_B2, tran_B3, tran_B4, tran_B5)
            out_S = self.decoder1(B_n[0], B_n[1], B_n[2], B_n[3], B_n[4])
            out_S_list.append(out_S)

            # prediction method (Equation 7)
            if n == self.loop-1:
                tmp = float(sum(range(self.loop+1)))
                out_Sf = sum([out_S_list[i]*(i+1)/tmp for i in range(len(out_S_list))])
                
        out_Pf = upsample_like(self.spv1(out_Sf), shape=shape)

        return torch.sigmoid(out_Pf[0])


class conv_bn_relu(nn.Module):
    def __init__(self, inc, outc, kernel, stride=1, padding=0, dilation=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(outc)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        return out

def upsample_like(src, tar=None, shape=None):
    if tar is not None:
        src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    elif tar is None and shape is not None:
        src = F.interpolate(src, size=shape, mode='bilinear', align_corners=True)
    return src