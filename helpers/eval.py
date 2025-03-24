"""
Helper functions for evaluation. 
Some parts taken from https://github.com/sauradip/STALE/blob/main/evaluation/utils_eval.py
"""

import logging
import numpy as np


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def wrapper_compute_average_precision(tiou_thresholds, activity_index, ground_truth, prediction):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(tiou_thresholds), len(activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = ground_truth.groupby('label')
        prediction_by_label = prediction.groupby('label')

        gt_present = list(ground_truth_by_label.groups.keys())
        pred_present = list(prediction_by_label.groups.keys())
        is_all_gt_present = len(gt_present) == len(activity_index)
        is_all_pred_present = len(pred_present) == len(activity_index)
        
        logging.info(f'All classes present GT subset? {is_all_gt_present} Pred subet? {is_all_pred_present}') 
        missed_pred_classes = []

        for i, cidx in enumerate(activity_index):
            if cidx not in gt_present:
                logging.warning(f'Class {cidx} not in GT subset.')
                continue

            if cidx not in pred_present:
                ap[:,cidx] = 0.0
                missed_pred_classes.append(cidx)
                logging.warning(f'Class {cidx} not in pred subset.')
                continue

            ground_truth_single_class_df = ground_truth_by_label.get_group(cidx).reset_index(drop=True)
            prediction_single_class_df = prediction_by_label.get_group(cidx).reset_index(drop=True)
            
            ap[:,cidx] = compute_average_precision_detection(
                ground_truth_single_class_df, prediction_single_class_df, tiou_thresholds
            )

        if len(missed_pred_classes) > 0:
            logging.info(f'Missed pred classes: {missed_pred_classes}')

        return ap


def evaluate(tiou_thresholds, activity_index, ground_truth, prediction):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        logging.info(f'# of predictions: {len(prediction)}. # of unique videos: {len(np.unique(prediction["video-id"]))}')
        
        ap = wrapper_compute_average_precision(tiou_thresholds, activity_index, ground_truth, prediction)
        
        gt_classes = ground_truth['label'].unique()
        if len(gt_classes) != len(activity_index):
            logging.warning('Some classes are missing in GT, so removing them from eval.')
            ap = ap[:, gt_classes]

        logging.info(f'tIoU thresholds: {tiou_thresholds}')
        
        # evaluate on 10 random samplings of x% of gt classes
        np.random.seed(1111)
        rmAP = np.zeros((10, len(ap)))
        for i in range(10):
            # TODO: change to 0.25 for evaluating zero-shot 75-25 split
            subset = np.random.choice(gt_classes, int(0.50 * len(gt_classes)), replace=False)
            rmAP[i] = ap[:, subset].mean(axis=1)
        rmAP = rmAP.mean(axis=0)
        average_rmAP = rmAP.mean()
        logging.info(f'Random subset mAP: {[float(round(m, 4)) for m in rmAP]}')
        logging.info(f'Random subset avg mAP: {average_rmAP:.4f}')

        mAP = ap.mean(axis=1)
        average_mAP = mAP.mean()
        logging.info(f'mAP: {[float(round(m, 4)) for m in mAP]}')
        logging.info(f'Average-mAP: {average_mAP:.4f}')
        
        return mAP, average_mAP


if __name__ == '__main__':
    pass