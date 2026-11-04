"""Background compensation of the SAM3 tracks (the step that wrote motion and mask-change residuals into tracks.jsonl).

Each frame registers the previous grayscale image to the current one with an affine transform estimated on background
corners (object masks excluded). Tracked objects then get motion_residual (centroid error after compensation, divided by
the image diagonal) and mask_change_residual (1 - IoU of the warped previous mask and the current mask). The selectors use
these residuals as motion events. Expects the BGR image (as read by cv2.imread when the training tracks were built).
"""
from __future__ import annotations


class BackgroundCompensation:
    """Past-to-current affine image registration; residuals are visual proxies only."""
    def __init__(self):
        self.previous = None
        self.objects = {}

    def update(self, image, objects):
        import cv2
        import numpy as np
        from pycocotools import mask as mask_utils
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        old = self.previous
        transform, inlier_fraction = None, None
        if old is not None:
            background = np.full(old.shape, 255, np.uint8)
            for obj in self.objects.values():
                rle = dict(obj["mask_rle"])
                rle["counts"] = rle["counts"].encode("ascii")
                mask = mask_utils.decode(rle).astype(np.uint8)
                background[cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0] = 0
            points = cv2.goodFeaturesToTrack(old, 180, 0.01, 7, mask=background)
            if points is not None and len(points) >= 12:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(old, gray, points, None)
                if tracked is not None:
                    valid = status.reshape(-1).astype(bool)
                    if valid.sum() >= 12:
                        cv2.setRNGSeed(0)
                        candidate, inliers = cv2.estimateAffinePartial2D(
                            points[valid], tracked[valid], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                        if candidate is not None and inliers is not None:
                            scale = float(np.linalg.norm(candidate[:, 0]))
                            inlier_fraction = float(inliers.mean())
                            if 0.75 <= scale <= 1.33 and inlier_fraction >= 0.6:
                                transform = candidate
        if transform is not None:
            height, width = gray.shape
            for obj in objects:
                previous = self.objects.get(obj["object_id"])
                if previous is None or previous.get("centroid_xy") is None or obj.get("centroid_xy") is None:
                    continue
                center = np.array([previous["centroid_xy"][0] * width,
                                   previous["centroid_xy"][1] * height, 1.0])
                predicted = transform @ center
                actual = np.array([obj["centroid_xy"][0] * width, obj["centroid_xy"][1] * height])
                obj["motion_residual"] = float(np.linalg.norm(actual - predicted) / np.hypot(width, height))
                rle_old, rle_new = dict(previous["mask_rle"]), dict(obj["mask_rle"])
                rle_old["counts"] = rle_old["counts"].encode("ascii")
                rle_new["counts"] = rle_new["counts"].encode("ascii")
                warp = cv2.warpAffine(mask_utils.decode(rle_old), transform, (width, height), flags=cv2.INTER_NEAREST) > 0
                current_mask = mask_utils.decode(rle_new) > 0
                union = np.count_nonzero(warp | current_mask)
                obj["mask_change_residual"] = float(1 - np.count_nonzero(warp & current_mask) / union) if union else 0.0
        self.previous = gray
        self.objects = {obj["object_id"]: obj for obj in objects}
        return {"background_alignment_valid": transform is not None,
                "background_inlier_fraction": inlier_fraction}
