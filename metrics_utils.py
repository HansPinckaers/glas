import numpy as np 
import scipy
from scipy.ndimage.measurements import label
import os

def hausdorff_distance(pred, true):

    haus1 = scipy.spatial.distance.directed_hausdorff(pred, true)[0]
    haus2 = scipy.spatial.distance.directed_hausdorff(true, pred)[0]
    haus_dist = max(haus1, haus2)

    return haus_dist

def find_nearest(mask, candidates):

    dst_map = scipy.ndimage.morphology.distance_transform_edt(1-mask)
    dst_map = dst_map.astype('int32')

    min_dst = 10000
    for idx in range(1, len(candidates)):
        candidate = candidates[idx]
        candidate = candidate.astype('int32')
        candidate *= dst_map
        dst = np.min(candidate[np.nonzero(candidate)])
        if dst < min_dst:
            min_dst = dst
            nearest_idx = idx
    return nearest_idx

def get_tp_fp_fn(pred, true):
    
    tp = 0
    fp = 0
    fn = 0

    true = np.copy(true) # get copy of GT
    pred = np.copy(pred) # get copy of prediction
    true = true.astype('int32')
    pred = pred.astype('int32')
    pred = label(pred)[0]
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    num_gt = true_id.copy()
    num_gt.remove(0)
    num_gt = len(num_gt)

    # Check to see whether there exists any predictions / GT instances
    if len(pred_id) == 1 and len(true_id) == 1:
        tp = 0
        fp = 0
        fn = 0
    elif len(pred_id) == 1:
        tp = 0
        fp = 0
        fn = len(true_id) - 1
    elif len(true_id) == 1:
        tp = 0
        fp = len(pred_id) -1
        fn = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for pred_idx in range(1, len(pred_id)):
        p_mask = pred_masks[pred_idx]
        true_pred_overlap = true[p_mask > 0]
        true_pred_overlap_id = np.unique(true_pred_overlap)
        true_pred_overlap_id = list(true_pred_overlap_id)
        try: # remove background
            true_pred_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(true_pred_overlap_id) > 0:
            t_mask_combine = np.zeros(p_mask.shape)
            for true_idx in true_pred_overlap_id:
                t_mask_combine += true_masks[true_idx]
            inter = (t_mask_combine * p_mask).sum()
            size_gt = t_mask_combine.sum()
            if inter / size_gt > 0.5:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = num_gt - tp
    
    return tp, fp, fn

def get_dice_info(pred, true):

    d_temp1 = 0
    d_temp2 = 0

    gamma = 1  #TODO this needs to be changed to give the appropriate weighting
    sigma = 1  #TODO this needs to be changed to give the appropriate weighting

    true = np.copy(true) # get copy of GT
    pred = np.copy(pred) # get copy of prediction
    true = true.astype('int32')
    pred = pred.astype('int32')
    pred = label(pred)[0]
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    num_gt = true_id.copy()
    num_gt.remove(0)
    num_gt = len(num_gt)

    # Check to see whether there exists any predictions / GT instances
    if len(pred_id) == 1 and len(true_id) == 1:
        tp = 0
        fp = 0
        fn = 0
    elif len(pred_id) == 1:
        tp = 0
        fp = 0
        fn = len(true_id) - 1
    elif len(true_id) == 1:
        tp = 0
        fp = len(pred_id) -1
        fn = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    
    # First term in RHS of object dice equation
    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try: # remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(pred_true_overlap_id) > 0:
            inter_max = 0
            for pred_idx in pred_true_overlap_id:
                inter = (pred_masks[pred_idx] * t_mask).sum()
                if inter > inter_max:
                    inter_max = inter
                    pred_idx_max = pred_idx
            p_mask_max = pred_masks[pred_idx_max]
            total = p_mask_max.sum() + t_mask.sum()
            d_temp1 += gamma * 2 * inter_max / total  # calculate dice (gamme gives more weight to larger glands in the GT)
    
    # Second term in RHS of object dice equation
    for pred_idx in range(1, len(pred_id)):
        p_mask = pred_masks[pred_idx]
        true_pred_overlap = true[p_mask > 0]
        true_pred_overlap_id = np.unique(true_pred_overlap)
        true_pred_overlap_id = list(true_pred_overlap_id)
        try: # remove background
            true_pred_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(true_pred_overlap_id) > 0:
            inter_max = 0
            for true_idx in true_pred_overlap_id:
                inter = (true_masks[true_idx] * p_mask).sum()
                if inter > inter_max:
                    inter_max = inter
                    true_idx_max = true_idx
            t_mask_max = true_masks[true_idx_max]
            total = t_mask_max.sum() + p_mask.sum()
            d_temp2 += sigma * 2 * inter_max / total  # calculate dice (gamme gives more weight to larger glands in the GT)
            
    return d_temp1, d_temp2


def get_haus_info(pred, true):

    h_temp1 = 0
    h_temp2 = 0

    gamma = 1  #TODO this needs to be changed to give the appropriate weighting
    sigma = 1  #TODO this needs to be changed to give the appropriate weighting

    true = np.copy(true) # get copy of GT
    pred = np.copy(pred) # get copy of prediction
    true = true.astype('int32')
    pred = pred.astype('int32')
    pred = label(pred)[0]
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    num_gt = true_id.copy()
    num_gt.remove(0)
    num_gt = len(num_gt)

    # Check to see whether there exists any predictions / GT instances
    if len(pred_id) == 1 and len(true_id) == 1:
        tp = 0
        fp = 0
        fn = 0
    elif len(pred_id) == 1:
        tp = 0
        fp = 0
        fn = len(true_id) - 1
    elif len(true_id) == 1:
        tp = 0
        fp = len(pred_id) -1
        fn = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    
    # First term in RHS of object hausdorff equation
    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try: # remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(pred_true_overlap_id) > 0:
            inter_max = 0
            for pred_idx in pred_true_overlap_id:
                inter = (pred_masks[pred_idx] * t_mask).sum()
                if inter > inter_max:
                    inter_max = inter
                    pred_idx_max = pred_idx
            p_mask_max = pred_masks[pred_idx_max]
            haus_dist = hausdorff_distance(p_mask_max, t_mask)
            h_temp1 += gamma * haus_dist  # calculate hausforff distance (gamme gives more weight to larger glands in the GT)
        else:
            nearest_idx = find_nearest(t_mask, pred_masks)  # if no matching predition, use closest prediction to corresponding GT
            p_mask_nearest = pred_masks[nearest_idx] 
            haus_dist = hausdorff_distance(p_mask_nearest, t_mask)
            h_temp1 += gamma * haus_dist  # calculate hausforff distance (gamme gives more weight to larger glands in the GT)
    
    # Second term in RHS of object hausdorff equation
    for pred_idx in range(1, len(pred_id)):
        p_mask = pred_masks[pred_idx]
        true_pred_overlap = true[p_mask > 0]
        true_pred_overlap_id = np.unique(true_pred_overlap)
        true_pred_overlap_id = list(true_pred_overlap_id)
        try: # remove background
            true_pred_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(true_pred_overlap_id) > 0:
            inter_max = 0
            for true_idx in true_pred_overlap_id:
                inter = (true_masks[true_idx] * p_mask).sum()
                if inter > inter_max:
                    inter_max = inter
                    true_idx_max = true_idx
            t_mask_max = true_masks[true_idx_max]
            haus_dist = hausdorff_distance(p_mask, t_mask_max)
            h_temp2 += sigma * haus_dist  # calculate hausforff distance (gamme gives more weight to larger glands in the GT)
        else:
            nearest_idx = find_nearest(p_mask, true_masks)  # if no matching GT, use closest GT to corresponding prediction
            t_mask_nearest = true_masks[nearest_idx] 
            haus_dist = hausdorff_distance(p_mask, t_mask_nearest)
            h_temp2 += sigma * haus_dist  # calculate hausforff distance (gamme gives more weight to larger glands in the GT)

    return h_temp1, h_temp2
