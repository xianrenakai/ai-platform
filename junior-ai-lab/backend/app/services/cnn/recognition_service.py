import base64
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import HTTPException, status
from PIL import Image

CNN_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "net_web" / "backend" / "ml_models"
_default_model = None
_model_load_error = None


def _get_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TensorFlow 不可用：{exc}") from exc


def _patch_input_layer(tf) -> None:
    orig = tf.keras.layers.InputLayer.from_config
    if getattr(orig, "_patched", False):
        return
    def patched(config):
        if isinstance(config, dict) and "batch_shape" in config and "batch_input_shape" not in config:
            config = {**config, "batch_input_shape": config.pop("batch_shape")}
        return orig(config)
    patched._patched = True
    tf.keras.layers.InputLayer.from_config = patched


def load_default_model() -> None:
    global _default_model, _model_load_error
    model_path = CNN_MODELS_DIR / "model.h5"
    if not model_path.exists():
        _default_model = None
        return
    try:
        tf = _get_tf()
        _patch_input_layer(tf)
        _default_model = tf.keras.models.load_model(str(model_path), compile=False)
        _default_model.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)
        _model_load_error = None
    except Exception as exc:
        _default_model = None
        _model_load_error = str(exc)


def _strip_header(b64: str) -> str:
    if "," in b64 and b64.strip().lower().startswith("data:image"):
        return b64.split(",", 1)[1]
    return b64


def preprocess_image(image_base64: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(_strip_header(image_base64))
        img = Image.open(BytesIO(img_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="图片数据无效") from exc
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape((1, 28, 28, 1))


def recognize_default(image_base64: str) -> dict:
    if _default_model is None:
        detail = f"识别服务暂不可用：{_model_load_error}" if _model_load_error else "识别服务暂不可用，请先上传默认模型"
        raise HTTPException(status_code=503, detail=detail)
    arr = preprocess_image(image_base64)
    preds = _default_model.predict(arr, verbose=0)
    return {"prediction": int(np.argmax(preds[0])), "confidence": float(np.max(preds[0]))}
