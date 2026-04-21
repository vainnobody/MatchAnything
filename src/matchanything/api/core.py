from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import cv2
import numpy as np
import torch

from ..hloc import logger, match_dense
from ..ui.utils import filter_matches, get_model


class ImageMatchingAPI(torch.nn.Module):
    default_conf = {
        "ransac": {
            "enable": True,
            "estimator": "opencv",
            "geometry": "homography",
            "method": "CV2_USAC_MAGSAC",
            "reproj_threshold": 3,
            "confidence": 0.9999,
            "max_iter": 10000,
        },
    }

    def __init__(
        self,
        conf: dict,
        device: str = "cpu",
        max_keypoints: int = 2000,
        match_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        if not conf.get("dense", False):
            raise RuntimeError("The lightweight runtime only supports dense MatchAnything models.")

        self.device = device
        self.conf = deepcopy({**self.default_conf, **conf})
        self.match_conf = deepcopy(self.conf["matcher"])
        self.match_conf["model"]["match_threshold"] = match_threshold
        self.match_conf["model"]["max_keypoints"] = max_keypoints
        self.matcher = get_model(self.match_conf)
        self.pred: Dict[str, Any] | None = None

    @torch.inference_mode()
    def forward(self, img0: np.ndarray, img1: np.ndarray) -> Dict[str, Any]:
        assert isinstance(img0, np.ndarray)
        assert isinstance(img1, np.ndarray)

        pred = match_dense.match_images(
            self.matcher,
            img0,
            img1,
            self.match_conf["preprocessing"],
            device=self.device,
        )
        if self.conf["ransac"]["enable"]:
            pred = filter_matches(
                pred,
                ransac_method=self.conf["ransac"]["method"],
                ransac_reproj_threshold=self.conf["ransac"]["reproj_threshold"],
                ransac_confidence=self.conf["ransac"]["confidence"],
                ransac_max_iter=self.conf["ransac"]["max_iter"],
            )
        self.pred = pred
        return pred

    @torch.inference_mode()
    def extract(self, image: np.ndarray, max_keypoints: int = 512, binarize: bool = False):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        detector = cv2.ORB_create(nfeatures=max_keypoints)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        keypoints_array = (
            np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        scores = (
            np.array([kp.response for kp in keypoints], dtype=np.float32)
            if keypoints
            else np.empty((0,), dtype=np.float32)
        )

        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)
        if binarize:
            descriptors = descriptors.astype(np.uint8)

        logger.info("Extracted %s ORB keypoints", len(keypoints_array))
        return {
            "keypoints": keypoints_array,
            "keypoints_orig": keypoints_array.copy(),
            "descriptors": descriptors,
            "scores": scores,
        }
