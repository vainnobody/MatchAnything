from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as F

from ...runtime import get_weights_dir
from ...vendor import ensure_vendor_imports
from .. import DEVICE, logger
from ..utils.base_model import BaseModel

ensure_vendor_imports()
from MatchAnything.src.config.default import get_cfg_defaults
from MatchAnything.src.lightning.lightning_loftr import PL_LoFTR


class MatchAnything(BaseModel):
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        self.conf = conf
        config = get_cfg_defaults()
        vendor_root = ensure_vendor_imports()

        if conf["model_name"] == "matchanything_eloftr":
            config_path = vendor_root / "configs" / "models" / "eloftr_model.py"
            config.merge_from_file(str(config_path))
            if config.LOFTR.COARSE.ROPE and config.DATASET.NPE_NAME == "megadepth":
                config.LOFTR.COARSE.NPE = [832, 832, conf["img_resize"], conf["img_resize"]]
        elif conf["model_name"] == "matchanything_roma":
            config_path = vendor_root / "configs" / "models" / "roma_model.py"
            config.merge_from_file(str(config_path))
            if str(DEVICE) == "cpu":
                config.LOFTR.FP16 = False
                config.ROMA.MODEL.AMP = False
        else:
            raise NotImplementedError(conf["model_name"])

        config.METHOD = conf["model_name"]
        config.LOFTR.MATCH_COARSE.THR = conf["match_threshold"]

        model_path = get_weights_dir(conf.get("models_dir")) / f"{conf['model_name']}.ckpt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model weights: {model_path}. "
                "Run `matchanything setup` or set MATCHANYTHING_MODELS_DIR."
            )

        self.net = PL_LoFTR(config, pretrained_ckpt=model_path, test_mode=True).matcher
        self.net.eval().to(DEVICE)
        logger.info("Loaded %s from %s", conf["model_name"], model_path)

    def _forward(self, data):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0).astype("uint8")
        img1 = img1.transpose(1, 2, 0).astype("uint8")
        img0_size, img1_size = np.array(img0.shape[:2]), np.array(img1.shape[:2])
        img0_gray = np.array(Image.fromarray(img0).convert("L"))
        img1_gray = np.array(Image.fromarray(img1).convert("L"))
        (img0_gray, hw0_new, mask0), (img1_gray, hw1_new, mask1) = map(
            lambda x: resize(x, df=32),
            [img0_gray, img1_gray],
        )

        img0_t = torch.from_numpy(img0_gray)[None][None] / 255.0
        img1_t = torch.from_numpy(img1_gray)[None][None] / 255.0
        batch = {"image0": img0_t, "image1": img1_t}
        batch.update(
            {
                "image0_rgb_origin": data["image0"],
                "image1_rgb_origin": data["image1"],
                "origin_img_size0": torch.from_numpy(img0_size)[None],
                "origin_img_size1": torch.from_numpy(img1_size)[None],
            }
        )

        if mask0 is not None:
            mask0_t = torch.from_numpy(mask0).to(DEVICE)
            mask1_t = torch.from_numpy(mask1).to(DEVICE)
            ts_mask_0, ts_mask_1 = F.interpolate(
                torch.stack([mask0_t, mask1_t], dim=0)[None].float(),
                scale_factor=0.125,
                mode="nearest",
                recompute_scale_factor=False,
            )[0].bool()
            batch.update({"mask0": ts_mask_0[None], "mask1": ts_mask_1[None]})

        batch = dict_to_cuda(batch, device=DEVICE)
        self.net(batch)
        mkpts0 = batch["mkpts0_f"].cpu()
        mkpts1 = batch["mkpts1_f"].cpu()
        mconf = batch["mconf"].cpu()

        if self.conf["model_name"] == "matchanything_eloftr":
            mkpts0 *= torch.tensor(hw0_new)[[1, 0]]
            mkpts1 *= torch.tensor(hw1_new)[[1, 0]]

        return {
            "keypoints0": mkpts0,
            "keypoints1": mkpts1,
            "mconf": mconf,
        }


def resize(img, resize=None, df=8, padding=True):
    w, h = img.shape[1], img.shape[0]
    w_new, h_new = process_resize(w, h, resize=resize, df=df, resize_no_larger_than=False)
    img_new = resize_image(img, (w_new, h_new), interp="pil_LANCZOS").astype("float32")
    h_scale, w_scale = img.shape[0] / img_new.shape[0], img.shape[1] / img_new.shape[1]
    mask = None
    if padding:
        img_new, mask = pad_bottom_right(img_new, max(h_new, w_new), ret_mask=True)
    return img_new, [h_scale, w_scale], mask


def process_resize(w, h, resize=None, df=None, resize_no_larger_than=False):
    if resize is not None:
        assert 0 < len(resize) <= 2
        if resize_no_larger_than and max(h, w) <= max(resize):
            w_new, h_new = w, h
        elif len(resize) == 1 and resize[0] > -1:
            scale = resize[0] / max(h, w)
            w_new, h_new = int(round(w * scale)), int(round(h * scale))
        elif len(resize) == 1 and resize[0] == -1:
            w_new, h_new = w, h
        else:
            w_new, h_new = resize[0], resize[1]
    else:
        w_new, h_new = w, h

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
    return w_new, h_new


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp_value = getattr(cv2, "INTER_" + interp[len("cv2_"):].upper())
        h, w = image.shape[:2]
        if interp_value == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp_value = cv2.INTER_LINEAR
        return cv2.resize(image, size, interpolation=interp_value)
    if interp.startswith("pil_"):
        interp_value = getattr(PIL.Image, interp[len("pil_"):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp_value)
        return np.asarray(resized, dtype=image.dtype)
    raise ValueError(f"Unknown interpolation {interp}.")


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:])
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[: inp.shape[0], : inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[: inp.shape[0], : inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, : inp.shape[1], : inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, : inp.shape[1], : inp.shape[2]] = True
        mask = mask[0]
    else:
        raise NotImplementedError
    return padded, mask


def dict_to_cuda(data_dict, device="cuda"):
    data_dict_cuda = {}
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict_cuda[key] = value.to(device)
        elif isinstance(value, dict):
            data_dict_cuda[key] = dict_to_cuda(value, device=device)
        elif isinstance(value, list):
            data_dict_cuda[key] = list_to_cuda(value, device=device)
        else:
            data_dict_cuda[key] = value
    return data_dict_cuda


def list_to_cuda(data_list, device="cuda"):
    data_list_cuda = []
    for obj in data_list:
        if isinstance(obj, torch.Tensor):
            data_list_cuda.append(obj.to(device))
        elif isinstance(obj, dict):
            data_list_cuda.append(dict_to_cuda(obj, device=device))
        elif isinstance(obj, list):
            data_list_cuda.append(list_to_cuda(obj, device=device))
        else:
            data_list_cuda.append(obj)
    return data_list_cuda
