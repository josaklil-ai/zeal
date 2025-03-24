"""
Helper functions for non-maximum suppression and interval generation.
"""
import numpy as np


def softnms_1d(
    intervals, scores, Nt=0.3, sigma=0.5, thresh=0.1, method='soft'
):
    indices = np.argsort(scores)[::-1]
    intervals = intervals[indices]
    scores = scores[indices]

    keep = []
    for i in range(len(scores)):
        if scores[i] < thresh:
            continue
        keep.append(i)
        ix1 = intervals[i, 0]
        ix2 = intervals[i, 1]

        for j in range(i+1, len(scores)):
            x1 = max(ix1, intervals[j, 0])
            x2 = min(ix2, intervals[j, 1])
            inter = max(0, x2 - x1 + 1)
            uni = (ix2 - ix1 + 1) + (intervals[j, 1] - intervals[j, 0] + 1) - inter
            overlap = inter / uni

            if method == 'linear':
                weight = 1 - overlap if overlap > Nt else 1
            elif method == 'soft':
                weight = np.exp(-(overlap**2) / sigma)
            else:  # original NMS
                weight = 0 if overlap > Nt else 1

            scores[j] *= weight

    return intervals[keep], scores[keep]


if __name__ == "__main__":
    pass