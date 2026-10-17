from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.models.cnn.models import (
    CnnDatasetRecord, CnnModelDesignRecord, CnnModelUploadRecord,
    CnnRaceRecord, CnnRecognitionRecord,
)
from app.services.cnn.ai_service import chat_with_ai, generate_evaluation_report

router = APIRouter(prefix="/api/cnn/evaluation", tags=["cnn-evaluation"])


def _score_dataset(r) -> int:
    if r is None: return 0
    cc = r.class_count or 0
    if cc >= 10: return 30
    if cc >= 5: return 22
    if cc >= 1: return 15
    return 0


def _score_design(r) -> int:
    if r is None: return 0
    cl = r.conv_layers or 0
    if cl >= 4: return 30
    if cl >= 2: return 22
    if cl >= 1: return 15
    return 0


def _score_test(upload, rec_count: int) -> int:
    if upload is None: return 0
    if rec_count <= 0: return 5
    acc = upload.final_val_accuracy or 0
    if acc >= 0.95: return 40
    if acc >= 0.85: return 30
    if acc >= 0.70: return 20
    return 10


def _get_evaluation(user: User, db: Session) -> dict:
    uid = user.id
    race = db.query(CnnRaceRecord).filter(CnnRaceRecord.user_id == uid).order_by(CnnRaceRecord.score.desc()).first()
    dataset = db.query(CnnDatasetRecord).filter(CnnDatasetRecord.user_id == uid).order_by(CnnDatasetRecord.created_at.desc()).first()
    design = db.query(CnnModelDesignRecord).filter(CnnModelDesignRecord.user_id == uid).order_by(CnnModelDesignRecord.created_at.desc()).first()
    upload = db.query(CnnModelUploadRecord).filter(CnnModelUploadRecord.user_id == uid).order_by(CnnModelUploadRecord.final_val_accuracy.desc()).first()
    rec_count = db.query(func.count(CnnRecognitionRecord.id)).filter(CnnRecognitionRecord.user_id == uid).scalar() or 0

    race_pts = min(20, int((race.score if race else 0) * 4))
    score = min(100.0, float(race_pts + _score_dataset(dataset) + _score_design(design) + _score_test(upload, rec_count)))
    completed = [n for n, obj in [("race", race), ("dataset", dataset), ("model_design", design), ("model_test", upload)] if obj is not None]
    missing = [n for n in ["race", "dataset", "model_design", "model_test"] if n not in completed]

    labels = {"race": "算术比赛活动", "dataset": "创建数据集活动", "model_design": "模型设计活动", "model_test": "模型测试活动"}
    if score >= 80:
        eval_text = "优秀！您在已完成的学习活动中表现出色。"
    elif score >= 60:
        eval_text = "良好！您已经掌握了部分核心知识。"
    else:
        eval_text = "继续努力！建议完成：" + "、".join(labels[n] for n in missing)

    return {
        "student": {"id": user.id, "username": user.username},
        "score": score,
        "score_breakdown": {"race": race_pts, "dataset": _score_dataset(dataset), "model_design": _score_design(design), "model_test": _score_test(upload, rec_count)},
        "completed_activities": completed,
        "completion_rate": round(len(completed) / 4 * 100, 1),
        "evaluation_text": eval_text,
        "best_model": {
            "accuracy": upload.final_val_accuracy or 0,
            "model_name": upload.model_filename,
            "conv_layers": upload.conv_layers or 0,
            "epochs": upload.epochs or 0,
        } if upload else None,
        "activity_stats": {
            "total": (
                (db.query(func.count(CnnRaceRecord.id)).filter(CnnRaceRecord.user_id == uid).scalar() or 0)
                + (db.query(func.count(CnnDatasetRecord.id)).filter(CnnDatasetRecord.user_id == uid).scalar() or 0)
                + (db.query(func.count(CnnModelDesignRecord.id)).filter(CnnModelDesignRecord.user_id == uid).scalar() or 0)
                + rec_count
            ),
            "race": db.query(func.count(CnnRaceRecord.id)).filter(CnnRaceRecord.user_id == uid).scalar() or 0,
            "dataset": db.query(func.count(CnnDatasetRecord.id)).filter(CnnDatasetRecord.user_id == uid).scalar() or 0,
            "model_design": db.query(func.count(CnnModelDesignRecord.id)).filter(CnnModelDesignRecord.user_id == uid).scalar() or 0,
            "model_train": rec_count,
            "model_test": rec_count,
        },
        "gains": ["理解了机器学习的基本工作流程", "掌握了卷积神经网络的基本结构", "体验了数据集对模型性能的重要性", "学会了如何评估和改进模型"],
        "suggestions": ["尝试更复杂的图像识别任务", "学习其他类型的神经网络", "了解深度学习在现实中的应用"],
        "improvements": [labels[n] for n in missing] or ["尝试创建更多样化的数据集", "设计更深层次的CNN模型"],
    }


@router.get("/me")
def get_my_evaluation(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_evaluation(current_user, db)


class ChatMessage(BaseModel):
    role: str
    content: str


class AiChatRequest(BaseModel):
    message: str = ""
    history: list[ChatMessage] = []


@router.post("/ai-report")
def get_ai_report(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return generate_evaluation_report(_get_evaluation(current_user, db))


@router.post("/ai-chat")
def ai_chat(
    body: AiChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return chat_with_ai(_get_evaluation(current_user, db), body.message, [m.model_dump() for m in body.history])
