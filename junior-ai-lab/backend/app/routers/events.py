from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import EventLog, User
from app.xlsx_simple import create_workbook

router = APIRouter(prefix="/api/events", tags=["events"])

MAX_EXPORT_ROWS = 2000


class EventCreate(BaseModel):
    module: str
    event_type: str
    activity_id: Optional[str] = None
    page: Optional[str] = None
    target: Optional[str] = None
    duration_ms: Optional[int] = None
    detail: dict[str, Any] = Field(default_factory=dict)


def _require_staff(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师或管理员可以查看行为记录")
    return current_user


def _require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可以删除行为记录")
    return current_user


def _parse_detail(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _apply_filters(
    query,
    current_user: User,
    module: Optional[str],
    activity_id: Optional[str],
    class_name: Optional[str],
    username: Optional[str],
    event_type: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
):
    # 老师只能看自己班级；admin 可按班级筛选
    if current_user.role == "teacher":
        query = query.filter(EventLog.class_name == current_user.class_name)
    elif class_name:
        query = query.filter(EventLog.class_name == class_name)

    if module:
        query = query.filter(EventLog.module == module)
    if activity_id:
        query = query.filter(EventLog.activity_id == activity_id)
    if username:
        query = query.filter(EventLog.username == username)
    if event_type:
        query = query.filter(EventLog.event_type == event_type)
    if start:
        query = query.filter(EventLog.created_at >= start)
    if end:
        query = query.filter(EventLog.created_at <= end)
    return query


def _serialize(row: EventLog) -> dict[str, Any]:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "username": row.username,
        "class_name": row.class_name,
        "module": row.module,
        "activity_id": row.activity_id,
        "event_type": row.event_type,
        "page": row.page,
        "target": row.target,
        "duration_ms": row.duration_ms,
        "detail": _parse_detail(row.detail),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.post("")
def report_event(
    body: EventCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """上报一条行为埋点；登录用户上报自己的行为，失败不阻断前端学习流程。"""
    db.add(
        EventLog(
            user_id=current_user.id,
            username=current_user.username,
            class_name=current_user.class_name,
            module=body.module,
            activity_id=body.activity_id,
            event_type=body.event_type,
            page=body.page,
            target=body.target,
            duration_ms=body.duration_ms,
            detail=json.dumps(body.detail, ensure_ascii=False),
        )
    )
    db.commit()
    return {"ok": True}


@router.get("")
def list_events(
    module: Optional[str] = Query(None),
    activity_id: Optional[str] = Query(None),
    class_name: Optional[str] = Query(None),
    username: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    limit: int = Query(500, ge=1, le=MAX_EXPORT_ROWS),
    db: Session = Depends(get_db),
    _staff: User = Depends(_require_staff),
):
    query = db.query(EventLog)
    query = _apply_filters(
        query, _staff, module, activity_id, class_name, username, event_type, start, end
    )
    rows = query.order_by(EventLog.created_at.desc()).limit(limit).all()
    return [_serialize(row) for row in rows]


@router.get("/export")
def export_events(
    module: Optional[str] = Query(None),
    activity_id: Optional[str] = Query(None),
    class_name: Optional[str] = Query(None),
    username: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    _staff: User = Depends(_require_staff),
):
    query = db.query(EventLog)
    query = _apply_filters(
        query, _staff, module, activity_id, class_name, username, event_type, start, end
    )
    rows = query.order_by(EventLog.created_at.asc()).limit(MAX_EXPORT_ROWS).all()

    header = ["时间", "用户ID", "用户名", "班级", "模块", "活动", "事件类型", "页面", "对象", "停留(ms)", "详情"]
    table: list[list[str]] = [header]
    for row in rows:
        detail = _parse_detail(row.detail)
        detail_text = json.dumps(detail, ensure_ascii=False) if detail else ""
        table.append(
            [
                row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else "",
                str(row.user_id),
                row.username or "",
                row.class_name or "",
                row.module or "",
                row.activity_id or "",
                row.event_type or "",
                row.page or "",
                row.target or "",
                "" if row.duration_ms is None else str(row.duration_ms),
                detail_text,
            ]
        )

    output = BytesIO(create_workbook(table))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=event-logs-{stamp}.xlsx"},
    )


@router.delete("/{event_id}")
def delete_event(
    event_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    row = db.get(EventLog, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    db.delete(row)
    db.commit()
    return {"ok": True}
