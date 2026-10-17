import json
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.models.cnn.models import CnnDatasetRecord
from app.services.cnn.dataset_service import generate_dataset

router = APIRouter(prefix="/api/cnn/dataset", tags=["cnn-dataset"])


class GenerateDatasetRequest(BaseModel):
    digits: list[int]
    counts: dict[str, int]


@router.get("/mine")
def list_my_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = (
        db.query(CnnDatasetRecord)
        .filter(CnnDatasetRecord.user_id == current_user.id)
        .order_by(CnnDatasetRecord.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "creator": current_user.username,
            "name": r.dataset_name,
            "createdAt": r.created_at,
            "digits": json.loads(r.digits or "[]"),
            "counts": json.loads(r.counts or "{}"),
            "totalImages": r.total_images,
            "classCount": r.class_count,
        }
        for r in records
    ]


@router.post("/generate")
def generate_dataset_zip(
    body: GenerateDatasetRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not body.digits:
        raise HTTPException(status_code=422, detail="digits 不能为空")

    zip_bytes, zip_filename, info = generate_dataset(body.digits, body.counts, current_user.username)

    db.add(CnnDatasetRecord(
        user_id=current_user.id,
        dataset_name=info["dataset_name"],
        digits=json.dumps(body.digits, ensure_ascii=False),
        counts=json.dumps(body.counts, ensure_ascii=False),
        total_images=info["total_images"],
        class_count=info["class_count"],
    ))
    db.commit()

    encoded = quote(zip_filename)
    return StreamingResponse(
        __import__("io").BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}; filename=dataset.zip"},
    )


@router.get("/{dataset_id}/download")
def download_existing_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(CnnDatasetRecord).filter(CnnDatasetRecord.id == dataset_id)
    if current_user.role != "admin":
        query = query.filter(CnnDatasetRecord.user_id == current_user.id)
    record = query.first()
    if record is None:
        raise HTTPException(status_code=404, detail="数据集记录不存在")

    digits = json.loads(record.digits or "[]")
    counts = json.loads(record.counts or "{}")
    zip_bytes, zip_filename, _ = generate_dataset(
        digits,
        counts,
        current_user.username,
        dataset_name=record.dataset_name,
    )
    encoded = quote(zip_filename)
    return StreamingResponse(
        __import__("io").BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}; filename=dataset.zip"},
    )
