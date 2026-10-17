import json
from math import ceil
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.models.cnn.models import (
    CnnDatasetRecord, CnnModelDesignRecord, CnnModelUploadRecord,
    CnnRaceRecord, CnnRecognitionRecord,
)

router = APIRouter(prefix="/api/cnn/statistics", tags=["cnn-statistics"])


def _fmt(dt) -> Optional[str]:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else None


def _loads(raw, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _dataset_score(r: Optional[CnnDatasetRecord]) -> int:
    if r is None: return 0
    cc = r.class_count or 0
    if cc >= 10: return 30
    if cc >= 5: return 22
    if cc >= 1: return 15
    return 0


def _design_score(r: Optional[CnnModelDesignRecord]) -> int:
    if r is None: return 0
    cl = r.conv_layers or 0
    if cl >= 4: return 30
    if cl >= 2: return 22
    if cl >= 1: return 15
    return 0


def _test_score(upload: Optional[CnnModelUploadRecord], rec_count: int) -> int:
    if upload is None: return 0
    if rec_count <= 0: return 5
    acc = upload.final_val_accuracy or 0
    if acc >= 0.95: return 40
    if acc >= 0.85: return 30
    if acc >= 0.70: return 20
    return 10


def _score_student(db: Session, user: User) -> dict:
    uid = user.id
    race = db.query(CnnRaceRecord).filter(CnnRaceRecord.user_id == uid).order_by(CnnRaceRecord.score.desc()).first()
    dataset = db.query(CnnDatasetRecord).filter(CnnDatasetRecord.user_id == uid).order_by(CnnDatasetRecord.created_at.desc()).first()
    design = db.query(CnnModelDesignRecord).filter(CnnModelDesignRecord.user_id == uid).order_by(CnnModelDesignRecord.created_at.desc()).first()
    upload = db.query(CnnModelUploadRecord).filter(CnnModelUploadRecord.user_id == uid).order_by(CnnModelUploadRecord.final_val_accuracy.desc()).first()
    rec_count = db.query(func.count(CnnRecognitionRecord.id)).filter(CnnRecognitionRecord.user_id == uid).scalar() or 0

    race_pts = min(20, int((race.score if race else 0) * 4))
    dataset_pts = _dataset_score(dataset)
    design_pts = _design_score(design)
    test_pts = _test_score(upload, rec_count)
    total = race_pts + dataset_pts + design_pts + test_pts

    completed = [name for name, obj in [("race", race), ("dataset", dataset), ("model_design", design), ("model_test", upload)] if obj is not None]
    times = [obj.created_at for _, obj in [("race", race), ("dataset", dataset), ("model_design", design), ("model_test", upload)] if obj is not None and obj.created_at]

    return {
        "race": race, "dataset": dataset, "design": design, "upload": upload,
        "recognition_count": rec_count,
        "score_breakdown": {"race": race_pts, "dataset": dataset_pts, "model_design": design_pts, "model_test": test_pts},
        "completed_activities": completed,
        "completed_count": len(completed),
        "score": float(total),
        "latest_time": max(times) if times else None,
    }


@router.get("/dataset")
def dataset_statistics(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    class_filter: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(User).filter(User.role == "student")
    if current_user.role == "student":
        q = q.filter(User.class_name == current_user.class_name)
    elif class_filter:
        q = q.filter(User.class_name == class_filter)
    if search:
        q = q.filter(User.username.contains(search))
    students = q.all()
    total_datasets = db.query(func.count(CnnDatasetRecord.id)).scalar() or 0
    active = db.query(func.count(func.distinct(CnnDatasetRecord.user_id))).scalar() or 0

    rows = []
    for s in students:
        recs = db.query(CnnDatasetRecord).filter(CnnDatasetRecord.user_id == s.id).order_by(CnnDatasetRecord.created_at.desc()).all()
        latest = recs[0].created_at if recs else None
        rows.append({
            "name": s.username, "dataset_count": len(recs), "has_data": bool(recs),
            "latest_time": _fmt(latest), "_latest": latest,
            "datasets": [{"dataset_name": r.dataset_name, "class_count": r.class_count, "total_images": r.total_images, "created_at": _fmt(r.created_at)} for r in recs],
        })

    rows.sort(key=lambda x: (0 if x["has_data"] else 1, -(x["_latest"].timestamp() if x["_latest"] else 0)))
    start = (page - 1) * page_size
    page_rows = rows[start:start + page_size]
    for r in page_rows: r.pop("_latest", None)

    return {
        "summary": {"total_students": len(students), "total_datasets": total_datasets, "active_students": active,
                    "avg_datasets": round(total_datasets / active, 1) if active else 0},
        "students": page_rows, "total": len(students), "page": page,
        "total_pages": ceil(len(students) / page_size) if students else 0,
    }


@router.get("/model")
def model_statistics(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    class_filter: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(User).filter(User.role == "student")
    if current_user.role == "student":
        q = q.filter(User.class_name == current_user.class_name)
    elif class_filter:
        q = q.filter(User.class_name == class_filter)
    if search:
        q = q.filter(User.username.contains(search))
    students = q.all()
    total_models = db.query(func.count(CnnModelDesignRecord.id)).scalar() or 0
    active = db.query(func.count(func.distinct(CnnModelDesignRecord.user_id))).scalar() or 0

    rows = []
    for s in students:
        models = db.query(CnnModelDesignRecord).filter(CnnModelDesignRecord.user_id == s.id).order_by(CnnModelDesignRecord.created_at.desc()).all()
        latest = models[0].created_at if models else None
        rows.append({
            "name": s.username, "model_count": len(models), "has_data": bool(models),
            "latest_time": _fmt(latest), "_latest": latest,
            "models": [{"model_name": m.model_name, "conv_layers": m.conv_layers, "pool_layers": m.pool_layers, "fc_neurons": m.fc_neurons, "created_at": _fmt(m.created_at)} for m in models],
        })

    rows.sort(key=lambda x: (0 if x["has_data"] else 1, -(x["_latest"].timestamp() if x["_latest"] else 0)))
    start = (page - 1) * page_size
    page_rows = rows[start:start + page_size]
    for r in page_rows: r.pop("_latest", None)

    return {
        "summary": {"total_students": len(students), "total_models": total_models, "active_students": active,
                    "avg_models": round(total_models / active, 1) if active else 0},
        "students": page_rows, "total": len(students), "page": page,
        "total_pages": ceil(len(students) / page_size) if students else 0,
    }


@router.get("/feedback")
def feedback_statistics(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    class_filter: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(User).filter(User.role == "student")
    if current_user.role == "student":
        q = q.filter(User.class_name == current_user.class_name)
    elif class_filter:
        q = q.filter(User.class_name == class_filter)
    if search:
        q = q.filter(User.username.contains(search))
    students = q.all()
    rows = []
    all_scores = []
    for s in students:
        stats = _score_student(db, s)
        score = stats["score"]
        all_scores.append(score)
        cc = stats["completed_count"]
        label = "优秀" if cc == 4 and score >= 80 else "良好" if cc >= 2 and score >= 60 else "需改进" if cc >= 1 else "未参与"
        dataset = stats["dataset"]
        design = stats["design"]
        upload = stats["upload"]
        rows.append({
            "name": s.username,
            "dataset": {
                "class_count": dataset.class_count,
                "total_images": dataset.total_images,
                "dataset_name": dataset.dataset_name,
            } if dataset else "未完成",
            "model_design": {
                "conv_layers": design.conv_layers,
                "pool_layers": design.pool_layers,
                "fc_neurons": design.fc_neurons,
                "output_neurons": design.output_neurons,
                "model_name": design.model_name,
            } if design else "未完成",
            "model_test": {
                "best_val_accuracy": upload.final_val_accuracy or upload.test_accuracy or upload.final_accuracy or 0,
                "recognition_count": stats["recognition_count"],
                "status": "已测试" if stats["recognition_count"] > 0 else "未测试",
            } if upload else "未完成",
            "completed_count": cc,
            "score": score,
            "label": label,
            "_latest": stats["latest_time"],
        })

    rows.sort(key=lambda x: (-x["completed_count"], -(x["_latest"].timestamp() if x["_latest"] else 0), -x["score"]))
    start = (page - 1) * page_size
    page_rows = rows[start:start + page_size]
    for r in page_rows: r.pop("_latest", None)

    completed_all = sum(1 for r in rows if r["completed_count"] == 4)
    return {
        "summary": {"total_students": len(students), "completed_all": completed_all,
                    "avg_score": round(sum(all_scores) / len(all_scores), 1) if all_scores else 0,
                    "total_feedback": sum(1 for r in rows if r["completed_count"] > 0)},
        "students": page_rows, "total": len(students), "page": page,
        "total_pages": ceil(len(students) / page_size) if students else 0,
    }
