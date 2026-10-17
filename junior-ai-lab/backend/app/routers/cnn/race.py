from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.models.cnn.models import CnnRaceRecord

router = APIRouter(prefix="/api/cnn/race", tags=["cnn-race"])


class RaceSaveRequest(BaseModel):
    total_questions: int
    correct_count: int
    wrong_count: int
    accuracy: float
    score: float


@router.post("/save", status_code=status.HTTP_201_CREATED)
def save_race(
    body: RaceSaveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.correct_count + body.wrong_count != body.total_questions:
        raise HTTPException(status_code=422, detail="correct_count + wrong_count 必须等于 total_questions")
    
    # 确保正确率不超过100%，并重新计算
    accuracy = min(body.correct_count / body.total_questions, 1.0) if body.total_questions > 0 else 0.0
    
    db.add(CnnRaceRecord(
        user_id=current_user.id,
        total_questions=body.total_questions,
        correct_count=body.correct_count,
        wrong_count=body.wrong_count,
        accuracy=accuracy,
        score=body.score,
    ))
    db.commit()
    return {"ok": True, "message": "比赛记录保存成功"}
