import io
import os
import random
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, status

# dataset_service.py lives under junior-ai-lab/backend/app/services/cnn.
# The real 60,000-image source is the repository-level net_web directory.
DATASET_SOURCE_DIR = Path(__file__).resolve().parents[5] / "net_web" / "backend" / "dataset_source"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _list_images(folder: Path) -> List[str]:
    return [f for f in os.listdir(folder) if Path(f).suffix.lower() in _IMAGE_EXTENSIONS]


def generate_dataset(
    digits: List[int],
    counts: Dict[str, int],
    username: str,
    dataset_name: Optional[str] = None,
) -> Tuple[bytes, str, dict]:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dataset_name = dataset_name or f"{username.strip() or 'user'}_{timestamp}"
    selected: dict[str, list[str]] = {}

    for digit in digits:
        key = str(digit)
        folder = DATASET_SOURCE_DIR / key
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"数字 {digit} 的源图片文件夹不存在")
        images = _list_images(folder)
        need = counts[key]
        if len(images) < need:
            raise HTTPException(status_code=400, detail=f"数字 {digit} 可用图片不足（需要 {need}，实际 {len(images)}）")
        selected[key] = random.sample(images, need)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, files in selected.items():
            for fname in files:
                zf.write(DATASET_SOURCE_DIR / key / fname, f"{dataset_name}/{key}/{fname}")

    distribution = {k: len(v) for k, v in selected.items()}
    info = {"dataset_name": dataset_name, "total_images": sum(distribution.values()), "class_count": len(digits), "class_distribution": distribution}
    return buf.getvalue(), f"{dataset_name}.zip", info
