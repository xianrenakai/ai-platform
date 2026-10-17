from math import ceil
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import asc, desc, func
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.models.cnn.models import CnnRaceRecord

router = APIRouter(prefix="/api/cnn/leaderboard", tags=["cnn-leaderboard"])


@router.get("")
def get_leaderboard(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    class_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 子查询：获取每个用户的最新记录
    subquery = (
        db.query(
            CnnRaceRecord.user_id,
            func.max(CnnRaceRecord.created_at).label("latest_time")
        )
        .group_by(CnnRaceRecord.user_id)
        .subquery()
    )

    # 基础查询：只获取每个用户的最新记录
    base_query = (
        db.query(CnnRaceRecord, User)
        .join(User, CnnRaceRecord.user_id == User.id)
        .join(
            subquery,
            (CnnRaceRecord.user_id == subquery.c.user_id) & 
            (CnnRaceRecord.created_at == subquery.c.latest_time)
        )
        .filter(User.role == "student")
    )

    # 学生只能看自己班级，教师可以通过class_filter筛选
    if current_user.role == "student":
        target_class = current_user.class_name
        base_query = base_query.filter(User.class_name == target_class)
    elif current_user.role == "teacher":
        base_query = base_query.filter(User.class_name == current_user.class_name)
    elif class_filter:
        base_query = base_query.filter(User.class_name == class_filter)

    # 排序：得分 > 正确率 > 题目数 > 时间
    base_query = base_query.order_by(
        desc(CnnRaceRecord.score),
        desc(CnnRaceRecord.accuracy),
        desc(CnnRaceRecord.total_questions),
        asc(CnnRaceRecord.created_at),
    )

    total = base_query.count()
    all_records = base_query.all()

    # 给所有记录编排名
    ranked_records = []
    for rank, (record, user) in enumerate(all_records, start=1):
        ranked_records.append((rank, record, user))

    # 学生模式：确保当前用户排在第一页第一个位置
    if current_user.role == "student":
        current_user_record = next(
            ((r, rec, u) for r, rec, u in ranked_records if u.id == current_user.id),
            None
        )
        other_records = [(r, rec, u) for r, rec, u in ranked_records if u.id != current_user.id]
        
        if current_user_record:
            if page == 1:
                # 第一页：当前用户 + 其他前9个
                page_records = [current_user_record] + other_records[:page_size - 1]
            else:
                # 其他页：跳过第一页已显示的9个其他记录
                start_idx = (page - 2) * page_size + (page_size - 1)
                page_records = other_records[start_idx:start_idx + page_size]
        else:
            # 当前用户没有记录，正常分页
            page_records = ranked_records[(page - 1) * page_size:page * page_size]
    else:
        # 教师模式：正常分页
        page_records = ranked_records[(page - 1) * page_size:page * page_size]

    entries = []
    for rank, record, user in page_records:
        # 确保正确率不超过100%
        # 强制重新计算：正确题数 / 完成题数
        accuracy = (record.correct_count / record.total_questions) if record.total_questions > 0 else 0.0
        accuracy = max(0.0, min(accuracy, 1.0))
        entries.append({
            "rank": rank,
            "student_name": user.display_name or user.username,
            "class_name": user.class_name or "未分配",
            "total_questions": record.total_questions,
            "correct_count": record.correct_count,
            "accuracy": accuracy,
            "score": record.score,
            "time": record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else "",
            "is_current_user": user.id == current_user.id,
        })

    # 获取所有班级列表（供教师使用）
    classes = []
    if current_user.role in {"teacher", "admin"}:
        class_query = db.query(User.class_name).filter(User.role == "student", User.class_name.isnot(None))
        if current_user.role == "teacher":
            class_query = class_query.filter(User.class_name == current_user.class_name)
        classes = [
            row[0] for row in 
            class_query
            .distinct()
            .all()
        ]

    return {
        "entries": entries,
        "total": total,
        "page": page,
        "total_pages": ceil(total / page_size) if total else 0,
        "current_user_class": current_user.class_name if current_user.role == "student" else None,
        "available_classes": classes,
    }
