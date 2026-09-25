"""Small, explicit inference and association helpers for development experiments."""
from copy import deepcopy
import math
from .evaluate import iou


def tile_boxes(width, height, size=960, overlap=0.25):
    if min(width, height, size) <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid image/tile dimensions or overlap')
    step = max(1, int(size * (1 - overlap)))

    def starts(length):
        last = max(0, length - size)
        return sorted(set([*range(0, last + 1, step), last]))

    return [[x, y, min(x + size, width), min(y + size, height)]
            for y in starts(height) for x in starts(width)]


def remap(rows, tile):
    x, y, right, bottom = tile
    result = []
    for row in rows:
        a, b, c, d = row['bbox_xyxy']
        box = [max(x, min(right, x+a)), max(y, min(bottom, y+b)),
               max(x, min(right, x+c)), max(y, min(bottom, y+d))]
        if box[2] > box[0] and box[3] > box[1]:
            result.append({**row, 'bbox_xyxy': box})
    return result


def suppress(rows, threshold=0.5, limit=300):
    kept = []
    for row in sorted(rows, key=lambda r: -r['confidence']):
        if all(iou(row['bbox_xyxy'], k['bbox_xyxy']) <= threshold for k in kept):
            kept.append(row)
        if len(kept) >= limit:
            break
    return kept


def predict(model, image, method, device='cpu', confidence=0.035):
    if method not in {'whole640', 'whole1280', 'tiles960'}:
        raise ValueError('Unknown fixed method')

    def infer(im, size):
        result = model.predict(im, imgsz=size, conf=confidence, iou=0.5,
                               max_det=300, rect=False, device=device, verbose=False)[0]
        return [{'bbox_xyxy': row[:4], 'confidence': row[4]}
                for row in result.boxes.data.cpu().tolist()]

    rows = infer(image, 1280 if method == 'whole1280' else 640)
    if method == 'tiles960':
        for tile in tile_boxes(*image.size):
            if tile == [0, 0, *image.size]:
                continue
            rows.extend(remap(infer(image.crop(tile), 640), tile))
        rows = suppress(rows)
    return rows


def transform_box(box, affine):
    points = [(box[0], box[1]), (box[2], box[1]),
              (box[2], box[3]), (box[0], box[3])]
    mapped = [(affine[0][0]*x+affine[0][1]*y+affine[0][2],
               affine[1][0]*x+affine[1][1]*y+affine[1][2]) for x,y in points]
    return [min(p[0] for p in mapped), min(p[1] for p in mapped),
            max(p[0] for p in mapped), max(p[1] for p in mapped)]


class Tracker:
    """Greedy one-to-one IoU tracker; not a learned tracker or SKU identifier."""
    def __init__(self, min_hits=3, max_missing=2, match_iou=0.3):
        if min_hits < 1 or max_missing < 0 or not 0 < match_iou <= 1:
            raise ValueError('Invalid tracking settings')
        self.min_hits, self.max_missing, self.match_iou = min_hits, max_missing, match_iou
        self.tracks, self.next_id = {}, 1

    def update(self, detections, affine=None):
        affine = affine if affine is not None else [[1, 0, 0], [0, 1, 0]]
        for track in self.tracks.values():
            track['bbox_xyxy'] = transform_box(track['bbox_xyxy'], affine)
            track['missing'] += 1
            track['observed'] = False
        pairs = sorted([(iou(t['bbox_xyxy'], d['bbox_xyxy']), tid, j)
                        for tid,t in self.tracks.items() for j,d in enumerate(detections)], reverse=True)
        used_tracks, used_detections = set(), set()
        for score, tid, j in pairs:
            if score < self.match_iou:
                break
            if tid in used_tracks or j in used_detections:
                continue
            track = self.tracks[tid]
            track.update(deepcopy(detections[j]), missing=0, observed=True, hits=track['hits']+1)
            used_tracks.add(tid)
            used_detections.add(j)
        self.tracks = {tid:t for tid,t in self.tracks.items() if t['missing'] <= self.max_missing}
        for j,d in enumerate(detections):
            if j not in used_detections:
                self.tracks[self.next_id] = {**deepcopy(d), 'id':self.next_id, 'missing':0, 'hits':1, 'observed':True}
                self.next_id += 1
        for t in self.tracks.values():
            t['confirmed'] = t['hits'] >= self.min_hits
        return deepcopy(list(self.tracks.values()))


def camera_motion(previous, current):
    """Estimate a guarded 2D affine camera motion using sparse optical flow."""
    import cv2
    import numpy as np
    identity = [[1., 0., 0.], [0., 1., 0.]]
    if previous is None:
        return identity, {'status':'first_frame'}
    points = cv2.goodFeaturesToTrack(previous, maxCorners=400, qualityLevel=.01, minDistance=8)
    if points is None or len(points) < 12:
        return identity, {'status':'insufficient_features'}
    target, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if target is None or status is None:
        return identity, {'status':'flow_failed'}
    good = status.ravel().astype(bool) & np.isfinite(target).all(axis=(1,2))
    if good.sum() < 12:
        return identity, {'status':'insufficient_matches'}
    matrix, inliers = cv2.estimateAffinePartial2D(points[good], target[good], method=cv2.RANSAC, ransacReprojThreshold=3)
    ratio = float(inliers.mean()) if inliers is not None else 0.
    if matrix is None or not np.isfinite(matrix).all():
        return identity, {'status':'fit_failed'}
    scale = math.hypot(matrix[0,0], matrix[1,0])
    if ratio < .5 or not .8 <= scale <= 1.2 or max(abs(matrix[0,2]), abs(matrix[1,2])) > max(current.shape)*.3:
        return identity, {'status':'fit_rejected', 'inlier_ratio':ratio}
    return matrix.tolist(), {'status':'accepted', 'inlier_ratio':ratio}
