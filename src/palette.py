





import colorsys

import numpy as np

KEYPOINTS = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
    "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee",
    "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
]

COLORS_RGB = np.array([
    [255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170],
    [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [255, 0, 170],
    [170, 0, 255], [255, 0, 255], [85, 0, 255],
], dtype=np.uint8)

PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]


MIRROR = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]

PART_GROUPS = {
    "head": [0, 14, 15, 16, 17],
    "neck_trunk": [1],
    "right_arm": [2, 3, 4],
    "left_arm": [5, 6, 7],
    "right_leg": [8, 9, 10],
    "left_leg": [11, 12, 13],
}


def palette_hues(bgr_swapped: bool = False) -> np.ndarray:

    cols = COLORS_RGB[:, ::-1] if bgr_swapped else COLORS_RGB
    hues = [colorsys.rgb_to_hsv(*(c / 255.0))[0] for c in cols]
    return np.array(hues, dtype=np.float32) * 2 * np.pi
