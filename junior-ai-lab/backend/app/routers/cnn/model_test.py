import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.models.cnn.models import CnnModelUploadRecord, CnnRecognitionRecord
from app.services.cnn.model_test_service import recognize_with_upload
from app.services.cnn.recognition_service import recognize_default

router = APIRouter(prefix="/api/cnn/model", tags=["cnn-model"])

CNN_UPLOADS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "net_web" / "backend" / "uploads"


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class RecognitionRequest(BaseModel):
    image: str


class UploadedRecognitionRequest(BaseModel):
    image: str
    model_upload_id: int


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_model(
    uploaded_file: UploadFile = File(..., alias="model_file"),
    class_file: UploadFile = File(...),
    traininfo_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not (uploaded_file.filename or "").lower().endswith(".h5"):
        raise HTTPException(status_code=422, detail="model_file 必须是 .h5 文件")

    async def _read_json(f: UploadFile, label: str) -> Tuple[dict, str]:
        if not (f.filename or "").lower().endswith(".json"):
            raise HTTPException(status_code=422, detail=f"{label} 必须是 .json 文件")
        try:
            data = json.loads((await f.read()).decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"{label} 不是有效的 JSON 文件") from exc
        return data, json.dumps(data, ensure_ascii=False)

    class_mapping, class_mapping_json = await _read_json(class_file, "class_file")
    traininfo, train_data_json = await _read_json(traininfo_file, "traininfo_file")

    # 从traininfo中提取model_info和dataset_info
    model_info = traininfo.get("model_info", {})
    dataset_info = traininfo.get("dataset_info", {})
    
    num_classes = dataset_info.get("num_classes") or len(class_mapping) or 10
    model_id = f"{num_classes}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    CNN_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CNN_UPLOADS_DIR / f"{model_id}.h5"
    async with aiofiles.open(model_path, "wb") as out:
        await out.write(await uploaded_file.read())

    record = CnnModelUploadRecord(
        user_id=current_user.id,
        model_id=model_id,
        model_filename=uploaded_file.filename,
        model_path=str(model_path),
        final_accuracy=_to_float(traininfo.get("final_accuracy")),
        final_val_accuracy=_to_float(traininfo.get("final_val_accuracy")),
        test_accuracy=_to_float(traininfo.get("test_accuracy")),
        epochs=_to_int(traininfo.get("epochs")),
        conv_layers=_to_int(model_info.get("conv_layers")),
        pool_layers=_to_int(model_info.get("pooling_layers")),
        dense_layers=_to_int(model_info.get("dense_layers")),
        total_params=_to_int(model_info.get("total_params")),
        train_data_json=train_data_json,
        class_mapping_json=class_mapping_json,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {
        "model_id": model_id,
        "model_upload_id": record.id,
        "model_info": {
            "model_name": uploaded_file.filename,
            "final_accuracy": record.final_accuracy,
            "final_val_accuracy": record.final_val_accuracy,
            "test_accuracy": record.test_accuracy,
            "epochs": record.epochs,
            "conv_layers": record.conv_layers,
            "pool_layers": record.pool_layers,
            "dense_layers": record.dense_layers,
            "total_params": record.total_params,
        },
    }


@router.post("/recognize")
def recognize(body: RecognitionRequest):
    return recognize_default(body.image)


@router.post("/recognize_uploaded")
def recognize_uploaded(
    body: UploadedRecognitionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    upload = db.query(CnnModelUploadRecord).filter(
        CnnModelUploadRecord.id == body.model_upload_id,
        CnnModelUploadRecord.user_id == current_user.id,
    ).first()
    if not upload:
        raise HTTPException(status_code=404, detail="模型上传记录不存在")
    result = recognize_with_upload(upload, body.image)
    db.add(CnnRecognitionRecord(
        user_id=current_user.id,
        model_upload_id=upload.id,
        predicted_class=str(result["prediction"]),
        confidence=float(result["confidence"]),
    ))
    db.commit()
    return result
