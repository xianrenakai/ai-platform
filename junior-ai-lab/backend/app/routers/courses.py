"""AI foundation course catalog, protected resource delivery and progress summary."""

import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.auth import decode_token, get_current_user
from app.database import get_db
from app.models import CourseEvaluation, Submission, User


router = APIRouter(prefix="/api/courses", tags=["courses"])

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESOURCE_DIR = BACKEND_DIR.parents[1] / "人工智能学习资源"
RESOURCE_DIR = Path(os.getenv("AI_COURSE_RESOURCE_DIR", str(DEFAULT_RESOURCE_DIR))).resolve()

COURSES = [
    {
        "id": "chapter-1", "order": 1, "title": "走进人工智能", "icon": "🤖",
        "subtitle": "认识人工智能的概念、发展历程与图灵测试",
        "description": "从生活中的人工智能现象出发，理解人工智能的基本概念，并沿着关键事件认识技术发展历程。",
        "objectives": ["能辨别生活中的人工智能应用", "了解人工智能发展的关键阶段", "理解图灵测试的基本思想"],
        "video_folder": "第一模块视频", "image_folder": "第一模块图片", "document_prefix": "第一模块",
        "quiz": {"question": "图灵测试主要用于考察什么？", "options": ["机器是否具有类似人类的智能表现", "计算机运行速度", "机器人的机械强度"], "answer": 0},
    },
    {
        "id": "chapter-2", "order": 2, "title": "人工智能应用", "icon": "🌐",
        "subtitle": "观察翻译、自动驾驶与人机博弈",
        "description": "分析真实应用场景，理解人工智能如何借助数据、算法和计算能力解决问题。",
        "objectives": ["说出人工智能的典型应用", "分析应用背后的输入、处理和输出", "认识人工智能能力的边界"],
        "video_folder": "第二模块视频", "image_folder": "第二模块图片", "document_prefix": "第二模块",
        "quiz": {"question": "人工智能系统完成任务通常离不开哪组三项基础条件？", "options": ["数据、算法与算力", "屏幕、鼠标与键盘", "文字、图片与声音"], "answer": 0},
    },
    {
        "id": "chapter-3", "order": 3, "title": "人工智能技术", "icon": "🎙️",
        "subtitle": "语音识别、语音合成与图像识别",
        "description": "认识声音和图像如何被计算机表示与处理。本课程独立讲解知识，不改变平台原有三个实验模块。",
        "objectives": ["区分语音识别与语音合成", "理解图像识别的一般流程", "能描述训练数据在识别任务中的作用"],
        "video_folder": "第三模块视频", "image_folder": "第三模块图片", "document_prefix": "第三模块",
        "quiz": {"question": "把人说的话转换成文字属于哪项技术？", "options": ["语音识别", "语音合成", "图像识别"], "answer": 0},
    },
    {
        "id": "chapter-4", "order": 4, "title": "人工智能算法与机器学习", "icon": "🧩",
        "subtitle": "理解算法、数据、算力与学习方式",
        "description": "通过实例认识算法和机器学习，比较监督学习与无监督学习的基本特点。",
        "objectives": ["理解算法的基本含义", "区分监督学习与无监督学习", "认识数据质量对模型结果的影响"],
        "video_folder": "第四模块视频", "image_folder": "第四模块图片", "document_prefix": "第四模块",
        "quiz": {"question": "使用带有正确标签的数据进行训练，通常称为什么？", "options": ["监督学习", "无监督学习", "随机搜索"], "answer": 0},
    },
    {
        "id": "chapter-5", "order": 5, "title": "人工智能伦理与安全", "icon": "🛡️",
        "subtitle": "关注隐私、偏见、信息安全与责任",
        "description": "结合网络案例识别人工智能使用中的安全风险，学习负责任地使用技术。",
        "objectives": ["识别隐私和信息安全风险", "理解算法偏见产生的原因", "形成负责任使用人工智能的意识"],
        "video_folder": "第五模块视频", "image_folder": "第五模块图片", "document_prefix": "第五模块",
        "quiz": {"question": "收到要求立即填写账号密码的陌生链接时，正确做法是什么？", "options": ["核实来源并拒绝直接填写", "立即填写避免账号失效", "转发给更多同学"], "answer": 0},
    },
    {
        "id": "chapter-6", "order": 6, "title": "人工智能的影响与发展", "icon": "🚀",
        "subtitle": "思考人工智能与未来学习、生活和职业",
        "description": "讨论人工智能带来的机会与挑战，在证据和合理想象的基础上表达自己的观点。",
        "objectives": ["分析人工智能带来的积极影响与挑战", "从多个角度评价技术应用", "形成面向未来的学习意识"],
        "video_folder": "第六模块视频", "image_folder": "第六模块图片", "document_prefix": "第六模块",
        "quiz": {"question": "面对人工智能的发展，我们更合适的态度是什么？", "options": ["主动学习并负责任地使用", "完全拒绝所有新技术", "认为人工智能可以解决一切问题"], "answer": 0},
    },
]

ALLOWED_SUFFIXES = {".mp4", ".png", ".jpg", ".jpeg", ".wav", ".doc", ".docx", ".pdf", ".ppt", ".pptx", ".txt", ".ipynb"}


def _resource_id(path: Path) -> str:
    relative = path.relative_to(RESOURCE_DIR).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]


def _resource_map() -> dict[str, Path]:
    if not RESOURCE_DIR.exists():
        return {}
    result: dict[str, Path] = {}
    for path in RESOURCE_DIR.rglob("*"):
        if not path.is_file() or ".ipynb_checkpoints" in path.parts or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        result[_resource_id(path)] = path
    return result


def _resources_for(course: dict, role: str) -> list[dict]:
    resources: list[dict] = []
    folders = [(course["video_folder"], "video"), (course["image_folder"], "image")]
    for folder_name, kind in folders:
        folder = RESOURCE_DIR / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
                resources.append({
                    "id": _resource_id(path), "name": path.stem, "filename": path.name,
                    "kind": kind, "size": path.stat().st_size, "downloadable": False,
                })
    # Original teaching files and notebooks are only exposed to teachers/admins.
    if role in {"teacher", "admin"}:
        teaching_dir = RESOURCE_DIR / "教学设计"
        candidates = list(teaching_dir.glob(f"{course['document_prefix']}*")) if teaching_dir.exists() else []
        candidates += list(RESOURCE_DIR.glob(f"{course['document_prefix']}*.ipynb"))
        for path in sorted(candidates, key=lambda item: item.name):
            if path.is_file():
                resources.append({
                    "id": _resource_id(path), "name": path.stem, "filename": path.name,
                    "kind": "notebook" if path.suffix.lower() == ".ipynb" else "document",
                    "size": path.stat().st_size, "downloadable": True,
                })
    return resources


def _latest_progress(db: Session, user_id: int) -> dict[str, dict]:
    rows = (
        db.query(Submission)
        .filter(Submission.user_id == user_id, Submission.module == "foundation")
        .order_by(Submission.created_at.desc())
        .all()
    )
    progress: dict[str, dict] = {}
    for row in rows:
        if row.activity_id not in progress:
            try:
                detail = json.loads(row.raw_data or "{}")
            except json.JSONDecodeError:
                detail = {}
            progress[row.activity_id] = {"score": row.score, "updated_at": row.created_at.isoformat(), "detail": detail}
    return progress


@router.get("")
def list_courses(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    progress = _latest_progress(db, current_user.id)
    items = []
    for course in COURSES:
        state = progress.get(course["id"], {})
        evaluation = db.query(CourseEvaluation).filter(CourseEvaluation.user_id == current_user.id, CourseEvaluation.course_id == course["id"]).first()
        teacher_score = float(evaluation.score) if evaluation else 0
        items.append({**course, "resources": _resources_for(course, current_user.role), "score": min(100, float(state.get("score", 0)) + teacher_score), "learning_score": state.get("score", 0), "teacher_score": teacher_score, "teacher_comment": evaluation.comment if evaluation else "", "progress": state.get("detail", {}), "updated_at": state.get("updated_at")})
    return {"courses": items, "resource_root_ready": RESOURCE_DIR.exists()}


@router.get("/summary")
def course_summary(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    progress = _latest_progress(db, current_user.id)
    scores = []
    for course in COURSES:
        learning = float(progress.get(course["id"], {}).get("score", 0))
        evaluation = db.query(CourseEvaluation).filter(CourseEvaluation.user_id == current_user.id, CourseEvaluation.course_id == course["id"]).first()
        scores.append(min(100, learning + (float(evaluation.score) if evaluation else 0)))
    completed = sum(1 for value in scores if value >= 60)
    return {"completed": completed, "total": len(COURSES), "percent": round(completed / len(COURSES) * 100), "average_score": round(sum(scores) / len(scores), 1)}


@router.get("/evaluations/students")
def evaluation_students(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="只有教师或管理员可以评价")
    query = db.query(User).filter(User.role == "student", User.is_active.is_(True))
    if current_user.role == "teacher":
        query = query.filter(User.class_name == current_user.class_name)
    students = query.order_by(User.class_name, User.username).all()
    evaluations = db.query(CourseEvaluation).filter(CourseEvaluation.user_id.in_([student.id for student in students])).all() if students else []
    lookup = {(item.user_id, item.course_id): item for item in evaluations}
    return [{
        "id": student.id, "username": student.username, "display_name": student.display_name or student.username,
        "class_name": student.class_name,
        "evaluations": {course["id"]: {"score": lookup[(student.id, course["id"])].score, "comment": lookup[(student.id, course["id"])].comment} for course in COURSES if (student.id, course["id"]) in lookup},
    } for student in students]


@router.put("/evaluations/students/{student_id}/{course_id}")
def save_evaluation(student_id: int, course_id: str, payload: dict, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="只有教师或管理员可以评价")
    if course_id not in {course["id"] for course in COURSES}:
        raise HTTPException(status_code=404, detail="课程不存在")
    student = db.get(User, student_id)
    if not student or student.role != "student":
        raise HTTPException(status_code=404, detail="学生不存在")
    if current_user.role == "teacher" and student.class_name != current_user.class_name:
        raise HTTPException(status_code=403, detail="只能评价自己班级的学生")
    score = max(0, min(10, float(payload.get("score", 0))))
    record = db.query(CourseEvaluation).filter(CourseEvaluation.user_id == student_id, CourseEvaluation.course_id == course_id).first()
    if record is None:
        record = CourseEvaluation(user_id=student_id, course_id=course_id, teacher_id=current_user.id)
        db.add(record)
    record.teacher_id = current_user.id
    record.score = score
    record.comment = str(payload.get("comment", "")).strip()[:1000]
    db.commit()
    return {"ok": True, "score": record.score, "comment": record.comment}


def _range_response(path: Path, media_type: str, range_header: str):
    size = path.stat().st_size
    try:
        units, requested = range_header.strip().split("=", 1)
        if units.lower() != "bytes" or "," in requested:
            raise ValueError
        start_raw, end_raw = requested.split("-", 1)
        start = int(start_raw) if start_raw else max(0, size - int(end_raw))
        end = int(end_raw) if end_raw else size - 1
        if start < 0 or end < start or start >= size:
            raise ValueError
        end = min(end, size - 1)
    except (ValueError, TypeError):
        raise HTTPException(status_code=416, detail="无效的文件分段范围")

    length = end - start + 1

    def iterator():
        remaining = length
        with path.open("rb") as source:
            source.seek(start)
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iterator(), status_code=206, media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(length),
        },
    )


@router.get("/resources/{resource_id}")
def get_resource(resource_id: str, request: Request, download: bool = Query(False), user: User = Depends(get_current_user)):
    path = _resource_map().get(resource_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="课程资源不存在")
    if path.suffix.lower() in {".doc", ".docx", ".ipynb", ".ppt", ".pptx"} and user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="该原始资料仅供教师下载")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")
    if range_header and not download:
        return _range_response(path, media_type, range_header)
    return FileResponse(path, media_type=media_type, filename=path.name if download else None, content_disposition_type="attachment" if download else "inline")
