import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import Submission, User

router = APIRouter(prefix="/api", tags=["submissions"])

MAX_SCORES = {
    "recognition": {
        "dictation": 100,
        "tech_exploration": 6,
        "concept_sorting": 6,
    },
    "synthesis": {
        "activity1": 6,
        "activity2": 6,
        "activity3": 8,
        "activity4": 10,
    },
}


def clamp_score(module: str, activity_id: str, score: float) -> float:
    max_score = MAX_SCORES.get(module, {}).get(activity_id)
    if max_score is None:
        return max(score, 0)
    return min(max(score, 0), max_score)



class SubmissionCreate(BaseModel):
    module: str
    activity_id: str
    score: float = 0
    raw_data: dict[str, Any] = Field(default_factory=dict)


class LeaderboardItem(BaseModel):
    rank: int
    user_id: int
    user_name: str
    score: float
    raw_data: dict[str, Any]
    created_at: str


@router.post("/submissions")
def create_submission(
    body: SubmissionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = (
        db.query(Submission)
        .filter(
            Submission.user_id == current_user.id,
            Submission.module == body.module,
            Submission.activity_id == body.activity_id,
        )
        .first()
    )
    raw_json = json.dumps(body.raw_data, ensure_ascii=False)
    if existing:
        existing.score = clamp_score(body.module, body.activity_id, body.score)
        existing.raw_data = raw_json
    else:
        db.add(
            Submission(
                user_id=current_user.id,
                module=body.module,
                activity_id=body.activity_id,
                score=clamp_score(body.module, body.activity_id, body.score),
                raw_data=raw_json,
            )
        )
    db.commit()
    return {"ok": True, "message": "成绩已保存"}


@router.get("/submissions/my")
def get_my_submissions(
    module: str = Query(...),
    activity_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(Submission).filter(
        Submission.user_id == current_user.id,
        Submission.module == module,
    )
    if activity_id:
        q = q.filter(Submission.activity_id == activity_id)
    rows = q.order_by(Submission.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "module": r.module,
            "activity_id": r.activity_id,
            "score": r.score,
            "raw_data": json.loads(r.raw_data or "{}"),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/submissions/count")
def get_submission_count(
    module: str = Query(...),
    activity_id: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    count = (
        db.query(Submission)
        .filter(
            Submission.user_id == current_user.id,
            Submission.module == module,
            Submission.activity_id == activity_id,
        )
        .count()
    )
    return {"count": count}


@router.get("/leaderboard", response_model=list[LeaderboardItem])
def get_leaderboard(
    module: str = Query(...),
    activity_id: str = Query(...),
    class_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 基础查询
    base_query = (
        db.query(Submission, User)
        .join(User, Submission.user_id == User.id)
        .filter(
            Submission.module == module,
            Submission.activity_id == activity_id,
            User.role == "student",
        )
    )
    
    # 学生只能看本班，教师可以通过class_filter筛选
    if current_user.role == "student":
        base_query = base_query.filter(User.class_name == current_user.class_name)
    elif current_user.role == "teacher":
        base_query = base_query.filter(User.class_name == current_user.class_name)
    elif class_filter:
        base_query = base_query.filter(User.class_name == class_filter)
    
    rows = base_query.order_by(Submission.score.desc(), Submission.created_at.asc()).all()
    
    # 获取所有班级列表（供教师使用）
    available_classes = []
    if current_user.role in {"teacher", "admin"}:
        available_classes = [
            row[0] for row in 
            db.query(User.class_name)
            .filter(User.role == "student", User.class_name.isnot(None))
            .distinct()
            .all()
        ]
    
    # 生成排行榜数据
    result: list[LeaderboardItem] = []
    for rank, (sub, user) in enumerate(rows, start=1):
        raw_data = json.loads(sub.raw_data or "{}")
        raw_data["user_class"] = user.class_name  # 添加班级信息
        raw_data["is_current_user"] = (user.id == current_user.id)
        
        result.append(
            LeaderboardItem(
                rank=rank,
                user_id=user.id,
                user_name=user.username,
                score=sub.score,
                raw_data=raw_data,
                created_at=sub.created_at.isoformat() if sub.created_at else "",
            )
        )
    
    # 学生模式：确保当前用户排在第一位
    if current_user.role == "student" and result:
        current_user_idx = next((i for i, item in enumerate(result) if item.user_id == current_user.id), None)
        if current_user_idx is not None and current_user_idx > 0:
            current_user_item = result.pop(current_user_idx)
            result.insert(0, current_user_item)
            # 重新计算排名
            for rank, item in enumerate(result, start=1):
                item.rank = rank if rank == 1 or item.user_id != current_user.id else current_user_idx + 1
    
    return result


@router.delete("/submissions/clear")
def clear_leaderboard(
    module: str = Query(...),
    activity_id: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师可清空排行榜")
    query = db.query(Submission).filter(
        Submission.module == module,
        Submission.activity_id == activity_id,
    )
    if current_user.role == "teacher":
        student_ids = db.query(User.id).filter(User.role == "student", User.class_name == current_user.class_name)
        query = query.filter(Submission.user_id.in_(student_ids))
    query.delete(synchronize_session=False)
    db.commit()
    return {"ok": True}
