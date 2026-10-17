from datetime import datetime
from io import BytesIO
import re
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user, hash_password
from app.database import get_db
from app.models import Submission, User
from app.models.cnn.models import (
    CnnDatasetRecord,
    CnnModelDesignRecord,
    CnnModelUploadRecord,
    CnnRaceRecord,
    CnnRecognitionRecord,
)
from app.xlsx_simple import create_workbook, read_first_sheet

router = APIRouter(prefix="/api/management", tags=["management"])


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    password: str = Field(default="123456", min_length=6)
    role: Literal["student", "teacher", "admin"] = "student"
    class_name: Optional[str] = None


class UserUpdate(BaseModel):
    username: Optional[str] = Field(default=None, min_length=1)
    display_name: Optional[str] = Field(default=None, min_length=1)
    password: Optional[str] = Field(default=None, min_length=1)
    role: Optional[Literal["student", "teacher", "admin"]] = None
    class_name: Optional[str] = None
    is_active: Optional[bool] = None


class ScoreCreate(BaseModel):
    user_id: int
    module: Literal["recognition", "synthesis"]
    activity_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0)
    raw_data: str = "{}"


class ScoreUpdate(BaseModel):
    module: Optional[Literal["recognition", "synthesis"]] = None
    activity_id: Optional[str] = Field(default=None, min_length=1)
    score: Optional[float] = Field(default=None, ge=0)
    raw_data: Optional[str] = None


ACTIVITY_LABELS = {
    "dictation": "听写挑战",
    "tech_exploration": "技术探究",
    "concept_sorting": "概念排序",
    "activity1": "活动一",
    "activity2": "活动二",
    "activity3": "活动三",
    "activity4": "活动四",
}


MAX_SCORES = {
    "recognition": {
        "dictation": 12,
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


def to_percent(module: str, activity_id: str, score: float) -> float:
    max_score = MAX_SCORES.get(module, {}).get(activity_id)
    if not max_score:
        return round(min(max(score, 0), 100), 1)
    return round(min(max(score, 0) / max_score * 100, 100), 1)


def require_staff(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师或管理员可以使用管理功能")
    return current_user


def _check_scope(operator: User, target: User) -> None:
    if operator.role == "admin":
        return
    if target.role != "student" or target.class_name != operator.class_name:
        raise HTTPException(status_code=403, detail="教师只能管理自己班级的学生")


def fmt_dt(value: Optional[datetime]) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else ""


def user_out(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name or user.username,
        "role": user.role,
        "class_name": user.class_name,
        "is_active": user.is_active,
        "must_change_password": user.must_change_password,
    }


@router.get("/students/template")
def student_import_template(_: User = Depends(require_staff)):
    output = BytesIO(create_workbook([
        ["学号", "姓名", "班级", "初始密码"],
        ["2026071101", "示例学生", "2026级七年级11班", "123456"],
    ]))
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=student-import-template.xlsx"},
    )


@router.post("/students/import")
async def import_students(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    if not (file.filename or "").lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="请上传 .xlsx 格式的学生名单")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="学生名单不能超过5MB")
    try:
        rows = read_first_sheet(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Excel 文件无法读取") from exc

    if not rows or tuple(str(v or "").strip() for v in rows[0][:4]) != ("学号", "姓名", "班级", "初始密码"):
        raise HTTPException(status_code=400, detail="模板表头不正确，请下载系统模板")

    pending = []
    errors = []
    seen = set()
    for row_number, row in enumerate(rows[1:], start=2):
        row = list(row) + [""] * (4 - len(row))
        student_id = str(row[0] or "").strip()
        display_name = str(row[1] or "").strip()
        class_name = str(row[2] or "").strip()
        password = str(row[3] or "123456").strip()
        if not any([student_id, display_name, class_name]):
            continue
        if not re.fullmatch(r"\d{10}", student_id):
            errors.append(f"第{row_number}行：学号必须是10位数字")
        elif not display_name:
            errors.append(f"第{row_number}行：姓名不能为空")
        elif current_user.role == "teacher" and class_name != current_user.class_name:
            errors.append(f"第{row_number}行：只能导入到教师负责班级 {current_user.class_name}")
        elif not class_name:
            errors.append(f"第{row_number}行：班级不能为空")
        elif len(password) < 6:
            errors.append(f"第{row_number}行：初始密码至少6位")
        elif student_id in seen or db.query(User).filter(User.username == student_id).first():
            errors.append(f"第{row_number}行：学号 {student_id} 已存在或重复")
        else:
            seen.add(student_id)
            pending.append((student_id, display_name, class_name, password))
    if errors:
        return {"imported": 0, "errors": errors[:100]}
    for student_id, display_name, class_name, password in pending:
        db.add(User(
            username=student_id, display_name=display_name,
            password_hash=hash_password(password), role="student",
            class_name=class_name, must_change_password=True, is_active=True,
        ))
    db.commit()
    return {"imported": len(pending), "errors": []}


def score_out(score: Submission, username: str = "") -> dict:
    return {
        "id": score.id,
        "user_id": score.user_id,
        "username": username,
        "module": score.module,
        "activity_id": score.activity_id,
        "activity_label": ACTIVITY_LABELS.get(score.activity_id, score.activity_id),
        "score": score.score,
        "percent_score": to_percent(score.module, score.activity_id, score.score),
        "raw_score": score.score,
        "raw_data": score.raw_data,
        "created_at": fmt_dt(score.created_at),
    }


@router.get("/users")
def list_users(
    role: Optional[Literal["student", "teacher", "admin"]] = None,
    keyword: str = "",
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    query = db.query(User)
    if current_user.role == "teacher":
        query = query.filter(User.role == "student", User.class_name == current_user.class_name)
    if role:
        query = query.filter(User.role == role)
    if keyword.strip():
        query = query.filter(User.username.contains(keyword.strip()))
    users = query.order_by(User.role.desc(), User.id.asc()).all()
    return [user_out(user) for user in users]


@router.post("/users")
def create_user(
    body: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    username = body.username.strip()
    display_name = body.display_name.strip()
    if body.role == "student" and not re.fullmatch(r"\d{10}", username):
        raise HTTPException(status_code=400, detail="学生登录账号必须是10位数字学号")
    if current_user.role == "teacher" and body.role != "student":
        raise HTTPException(status_code=403, detail="教师只能创建学生账号")
    target_class = current_user.class_name if current_user.role == "teacher" else body.class_name
    if body.role != "admin" and not target_class:
        raise HTTPException(status_code=400, detail="请选择班级")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="该用户名已存在")
    user = User(
        username=username,
        display_name=display_name,
        password_hash=hash_password(body.password),
        role=body.role,
        class_name=target_class,
        must_change_password=True,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user_out(user)


@router.put("/users/{user_id}")
def update_user(
    user_id: int,
    body: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    _check_scope(current_user, user)
    if current_user.role == "teacher" and (body.username is not None or body.role is not None or body.class_name is not None):
        raise HTTPException(status_code=403, detail="教师不能修改学号、角色或班级")
    if body.username is not None:
        username = body.username.strip()
        exists = db.query(User).filter(User.username == username, User.id != user_id).first()
        if exists:
            raise HTTPException(status_code=400, detail="该用户名已存在")
        user.username = username
    if body.display_name is not None:
        user.display_name = body.display_name.strip()
    if body.password:
        user.password_hash = hash_password(body.password)
        user.must_change_password = True
    if body.role:
        user.role = body.role
    if body.class_name is not None:
        user.class_name = body.class_name
    if body.is_active is not None:
        user.is_active = body.is_active
    db.commit()
    db.refresh(user)
    return user_out(user)


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能删除当前登录用户")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    _check_scope(current_user, user)

    db.query(Submission).filter(Submission.user_id == user_id).delete(synchronize_session=False)
    db.query(CnnRecognitionRecord).filter(CnnRecognitionRecord.user_id == user_id).delete(synchronize_session=False)
    db.query(CnnRaceRecord).filter(CnnRaceRecord.user_id == user_id).delete(synchronize_session=False)
    db.query(CnnDatasetRecord).filter(CnnDatasetRecord.user_id == user_id).delete(synchronize_session=False)
    db.query(CnnModelDesignRecord).filter(CnnModelDesignRecord.user_id == user_id).delete(synchronize_session=False)
    db.query(CnnModelUploadRecord).filter(CnnModelUploadRecord.user_id == user_id).delete(synchronize_session=False)
    db.delete(user)
    db.commit()
    return {"ok": True}


@router.get("/scores")
def list_scores(
    user_id: Optional[int] = None,
    module: Optional[Literal["recognition", "synthesis"]] = None,
    keyword: str = "",
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    query = db.query(Submission, User.username).join(User, Submission.user_id == User.id)
    if current_user.role == "teacher":
        query = query.filter(User.class_name == current_user.class_name, User.role == "student")
    if user_id:
        query = query.filter(Submission.user_id == user_id)
    if module:
        query = query.filter(Submission.module == module)
    if keyword.strip():
        query = query.filter(User.username.contains(keyword.strip()))
    rows = query.order_by(Submission.created_at.desc()).limit(limit).all()
    return [score_out(score, username) for score, username in rows]


@router.post("/scores")
def create_score(
    body: ScoreCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    user = db.get(User, body.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    _check_scope(current_user, user)
    score = Submission(
        user_id=body.user_id,
        module=body.module,
        activity_id=body.activity_id,
        score=body.score,
        raw_data=body.raw_data or "{}",
    )
    db.add(score)
    db.commit()
    db.refresh(score)
    return score_out(score, user.username)


@router.put("/scores/{score_id}")
def update_score(
    score_id: int,
    body: ScoreUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    score = db.get(Submission, score_id)
    if not score:
        raise HTTPException(status_code=404, detail="成绩记录不存在")
    target_user = db.get(User, score.user_id)
    if target_user:
        _check_scope(current_user, target_user)
    if body.module:
        score.module = body.module
    if body.activity_id:
        score.activity_id = body.activity_id
    if body.score is not None:
        score.score = body.score
    if body.raw_data is not None:
        score.raw_data = body.raw_data or "{}"
    db.commit()
    db.refresh(score)
    user = db.get(User, score.user_id)
    return score_out(score, user.username if user else "")


@router.delete("/scores/{score_id}")
def delete_score(
    score_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_staff),
):
    score = db.get(Submission, score_id)
    if not score:
        raise HTTPException(status_code=404, detail="成绩记录不存在")
    target_user = db.get(User, score.user_id)
    if target_user:
        _check_scope(current_user, target_user)
    db.delete(score)
    db.commit()
    return {"ok": True}
