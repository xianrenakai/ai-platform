import json
import time
from typing import Optional

import numpy as np
from fastapi import HTTPException

from app.models.cnn.models import CnnModelUploadRecord
from app.services.cnn.recognition_service import preprocess_image

_model_cache: dict = {}
MAX_CACHE = 5
_tf_load_error: Optional[str] = None

_KERAS3_REMOVE_KEYS = frozenset({
    "registered_name",
    "module",
    "build_config",
    "compile_config",
    "quantization_config",
})
_INPUT_LAYER_REMOVE = frozenset({"optional", "ragged", "sparse", "registered_name", "module"})


def _get_tf():
    global _tf_load_error
    try:
        import tensorflow as tf
        _tf_load_error = None
        return tf
    except Exception as exc:
        _tf_load_error = str(exc)
        return None


def _simplify_dtype(value):
    """
    Keras 3 serializes dtype as:
      {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, ...}
    TF 2.15 only accepts a plain string like 'float32'.
    """
    if isinstance(value, dict) and value.get("class_name") == "DTypePolicy":
        inner = value.get("config", {})
        return inner.get("name", "float32")
    return value


def _clean_config(config):
    """
    Recursively clean a layer config dict:
      1. Replace DTypePolicy objects with plain dtype strings.
      2. Strip Keras-3-only top-level keys.
      3. Strip 'registered_name' / 'module' from nested serialized objects.
    """
    if not isinstance(config, dict):
        return config

    result = {}
    for key, value in config.items():
        if key in _KERAS3_REMOVE_KEYS:
            continue
        if key == "dtype":
            result[key] = _simplify_dtype(value)
        elif isinstance(value, dict):
            # Nested serialized Keras object — clean it recursively
            result[key] = _clean_config(value)
        elif isinstance(value, list):
            result[key] = [_clean_config(v) if isinstance(v, dict) else v for v in value]
        else:
            result[key] = value
    return result


def _patch_all_layers(tf) -> None:
    base_cls = tf.keras.layers.Layer
    if getattr(base_cls.from_config, "_compat_patched", False):
        return

    original_func = base_cls.from_config.__func__

    @classmethod  # type: ignore[misc]
    def patched_from_config(cls, config):
        if isinstance(config, dict):
            config = _clean_config(config)
        return original_func(cls, config)

    patched_from_config._compat_patched = True
    base_cls.from_config = patched_from_config


def _patch_input_layer(tf) -> None:
    cls = tf.keras.layers.InputLayer
    if getattr(cls.from_config, "_input_patched", False):
        return

    outer = cls.from_config.__func__

    @classmethod  # type: ignore[misc]
    def patched(cls2, config):
        if isinstance(config, dict):
            config = _clean_config(config)
            config = {k: v for k, v in config.items() if k not in _INPUT_LAYER_REMOVE}
            if "batch_shape" in config and "batch_input_shape" not in config:
                config["batch_input_shape"] = config.pop("batch_shape")
            else:
                config.pop("batch_shape", None)
        return outer(cls2, config)

    patched._input_patched = True
    cls.from_config = patched


def _load_model_compat(tf, model_path: str):
    _patch_all_layers(tf)
    _patch_input_layer(tf)
    return tf.keras.models.load_model(model_path, compile=False)


def _heuristic_recognize(arr: np.ndarray, class_mapping: dict, model_id: str) -> dict:
    image = arr.reshape(28, 28)
    dark = image < 0.8
    if not np.any(dark):
        idx, confidence = 0, 0.1
    else:
        ys, xs = np.where(dark)
        width = max(1, int(xs.max() - xs.min() + 1))
        height = max(1, int(ys.max() - ys.min() + 1))
        density = float(dark.mean())
        aspect = height / width
        if aspect > 2.2 and density < 0.22:
            idx, confidence = 1, 0.72
        elif density > 0.35:
            idx, confidence = 8, 0.55
        elif width > height * 1.2:
            idx, confidence = 7, 0.50
        else:
            idx, confidence = 0, 0.45
    return {
        "prediction": class_mapping.get(str(idx), str(idx)),
        "confidence": confidence,
        "model_id": model_id,
        "fallback": True,
        "message": f"TensorFlow 不可用，已使用兜底识别。原因：{_tf_load_error or '未知'}",
    }


def get_or_load_model(record: CnnModelUploadRecord) -> tuple:
    tf = _get_tf()
    if tf is None:
        class_mapping = json.loads(record.class_mapping_json or "{}")
        return None, class_mapping

    cached = _model_cache.get(record.model_id)
    if cached:
        cached["last_used"] = time.time()
        return cached["model"], cached["class_mapping"]

    if len(_model_cache) >= MAX_CACHE:
        oldest = min(_model_cache, key=lambda k: _model_cache[k]["last_used"])
        del _model_cache[oldest]

    try:
        model = _load_model_compat(tf, record.model_path)
        class_mapping = json.loads(record.class_mapping_json or "{}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型加载失败：{exc}") from exc

    _model_cache[record.model_id] = {
        "model": model,
        "class_mapping": class_mapping,
        "last_used": time.time(),
    }
    return model, class_mapping


def recognize_with_upload(record: CnnModelUploadRecord, image_base64: str) -> dict:
    model, class_mapping = get_or_load_model(record)
    arr = preprocess_image(image_base64)
    if model is None:
        return _heuristic_recognize(arr, class_mapping, record.model_id)

    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    return {
        "prediction": class_mapping.get(str(idx), str(idx)),
        "confidence": float(np.max(preds[0])),
        "model_id": record.model_id,
    }
