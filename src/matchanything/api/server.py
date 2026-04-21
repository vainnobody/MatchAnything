from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from ..config import load_runtime_config
from ..hloc import DEVICE
from ..ui.utils import get_matcher_zoo
from .. import __version__
from . import ImagesInput, to_base64_nparray
from .core import ImageMatchingAPI


class RuntimeService:
    def __init__(self, matcher_zoo: Dict[str, dict], default_matcher: str, device: str):
        self.matcher_zoo = matcher_zoo
        self.default_matcher = default_matcher
        self.device = device
        self._apis: Dict[str, ImageMatchingAPI] = {}

    def get_api(self, matcher_name: str) -> ImageMatchingAPI:
        if matcher_name not in self.matcher_zoo:
            raise KeyError(f"Unknown matcher `{matcher_name}`")
        if matcher_name not in self._apis:
            self._apis[matcher_name] = ImageMatchingAPI(
                conf=deepcopy(self.matcher_zoo[matcher_name]),
                device=self.device,
            )
        return self._apis[matcher_name]

    def load_image(self, file_path: Union[str, UploadFile]) -> np.ndarray:
        source = Path(file_path).resolve(strict=False) if isinstance(file_path, str) else file_path.file
        with Image.open(source) as img:
            return np.array(img.convert("RGB"))


def _to_jsonable(output: dict, skip_keys: set[str] | None = None) -> dict:
    skip = skip_keys or set()
    payload = {}
    for key, value in output.items():
        if key in skip:
            continue
        if hasattr(value, "tolist"):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    return payload


def create_app(config: str | Path | None = None, default_matcher: str = "matchanything_eloftr") -> FastAPI:
    cfg = load_runtime_config(config)
    matcher_zoo = get_matcher_zoo(cfg["matcher_zoo"])
    service = RuntimeService(matcher_zoo=matcher_zoo, default_matcher=default_matcher, device=str(DEVICE))
    app = FastAPI(title="MatchAnything API", version=__version__)

    @app.get("/")
    def root():
        return {"name": "MatchAnything", "version": __version__}

    @app.get("/version")
    def version():
        return {"version": __version__}

    @app.post("/v1/match")
    async def match(
        image0: UploadFile = File(...),
        image1: UploadFile = File(...),
        matcher: str = Form(default_matcher),
    ):
        try:
            api = service.get_api(matcher)
            image0_array = service.load_image(image0)
            image1_array = service.load_image(image1)
            output = api(image0_array, image1_array)
            return JSONResponse(content=_to_jsonable(output, {"image0_orig", "image1_orig"}))
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @app.post("/v1/extract")
    async def extract(input_info: ImagesInput):
        try:
            preds = []
            max_keypoints = input_info.max_keypoints or [512] * len(input_info.data)
            api = service.get_api(default_matcher)
            for index, input_image in enumerate(input_info.data):
                image_array = to_base64_nparray(input_image)
                output = api.extract(
                    image_array,
                    max_keypoints=max_keypoints[index],
                    binarize=input_info.binarize,
                )
                preds.append(_to_jsonable(output))
            return JSONResponse(content=preds)
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    return app
