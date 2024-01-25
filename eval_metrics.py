SIGMOID_THRESH = 0.5

import numpy as np
from sklearn.metrics import auc

np.seterr(invalid='ignore')

def sigmoid_confused_matrix(pred_logit, raw_label, num_classes, thresh):
    assert pred_logit.shape[0] == num_classes - 1

    class_p = np.zeros((num_classes,), dtype=np.float64)
    class_tp = np.zeros((num_classes,), dtype=np.float64)
    class_fn = np.zeros((num_classes,), dtype=np.float64)

    for i in range(1, num_classes):
        pred = pred_logit[i - 1] > thresh
        label = raw_label == i
        class_tp[i] = np.sum(label & pred)
        class_p[i] = np.sum(pred)
        class_fn[i] = np.sum(label) - class_tp[i]

    return class_p, class_tp, class_fn


def sigmoid_metrics(results, gt_seg_maps, num_classes, compute_aupr=False):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs

    if compute_aupr:
        threshs = np.linspace(0, 1, 11)  # 0.1
    else:
        threshs = [SIGMOID_THRESH]

    total_p_list = []
    total_tp_list = []
    total_fn_list = []

    for thresh in threshs:
        total_p = np.zeros((num_classes,), dtype=np.float64)
        total_tp = np.zeros((num_classes,), dtype=np.float64)
        total_fn = np.zeros((num_classes,), dtype=np.float64)

        for i in range(num_imgs):
            if isinstance(results[i], tuple):
                result = results[i][0]
            else:
                result = results[i]

            p, tp, fn = sigmoid_confused_matrix(result, gt_seg_maps[i], num_classes, thresh)
            total_p += p
            total_tp += tp
            total_fn += fn

        total_p_list.append(total_p)
        total_tp_list.append(total_tp)
        total_fn_list.append(total_fn)

    if len(threshs) > 1:
        index = int(np.argmax(threshs == 0.5))
    else:
        index = 0
    total_p = total_p_list[index]
    total_tp = total_tp_list[index]
    total_fn = total_fn_list[index]

    maupr = np.zeros((num_classes,), dtype=np.float64)
    total_p_list = np.stack(total_p_list)
    total_tp_list = np.stack(total_tp_list)
    total_fn_list = np.stack(total_fn_list)

    ppv_list = np.nan_to_num(total_tp_list / total_p_list, nan=1)
    s_list = np.nan_to_num(total_tp_list / (total_tp_list + total_fn_list), nan=0)

    if compute_aupr:
        for i in range(1, len(maupr)):
            x = s_list[:, i]
            y = ppv_list[:, i]
            maupr[i] = auc(x, y)

    return total_p, total_tp, total_fn, maupr


def metrics(results, gt_seg_maps, num_classes, nan_to_num=None):

    compute_aupr = True

    total_p, total_tp, total_fn, maupr = sigmoid_metrics(results, gt_seg_maps, num_classes, compute_aupr)

    dice = 2 * total_tp / (total_p + total_tp + total_fn)
    iou = total_tp / (total_p + total_fn)

    if nan_to_num is not None:
        return np.nan_to_num(iou, nan=nan_to_num), \
               np.nan_to_num(dice, nan=nan_to_num), \
               np.nan_to_num(maupr, nan=nan_to_num)
    else:
        return maupr, dice, iou

