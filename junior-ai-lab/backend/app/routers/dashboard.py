from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import Module, Submission, User
from app.models.cnn.models import (
    CnnDatasetRecord, CnnModelDesignRecord, CnnModelUploadRecord,
    CnnRaceRecord,
)

router = APIRouter(prefix="/api", tags=["dashboard"])


class ModuleInfo(BaseModel):
    id: int
    module_key: str
    module_name: str
    description: str
    icon: str
    route_path: str

    class Config:
        from_attributes = True


class ActivityScore(BaseModel):
    activity_id: str
    score: float
    created_at: Optional[str] = None


class ModuleSummary(BaseModel):
    module_key: str
    module_name: str
    activities: List[ActivityScore]
    avg_score: float


class StudentSummary(BaseModel):
    user_id: int
    username: str
    class_name: Optional[str]
    modules: Dict[str, float]
    plain_password: Optional[str] = None


class DashboardResponse(BaseModel):
    user: Dict[str, Any]
    modules: List[ModuleSummary]
    radar: Dict[str, Any]
    registered_modules: List[ModuleInfo]
    students_summary: Optional[List[StudentSummary]] = None


RECOGNITION_DIMS = ["听写挑战", "技术探究", "概念排序"]
SYNTHESIS_DIMS = ["活动一", "活动二", "活动三", "活动四"]
CNN_DIMS = ["算术比赛", "数据集", "模型设计", "模型测试"]

RECOGNITION_ACTS = ["dictation", "tech_exploration", "concept_sorting"]
SYNTHESIS_ACTS = ["activity1", "activity2", "activity3", "activity4"]
CNN_ACTS = ["race", "dataset", "model_design", "model_test"]

RECOGNITION_MAX_SCORES = {
    "dictation": 12,
    "tech_exploration": 6,
    "concept_sorting": 6,
}

SYNTHESIS_MAX_SCORES = {
    "activity1": 6,
    "activity2": 6,
    "activity3": 8,
    "activity4": 10,
}


def _to_percent(module: str, activity_id: str, score: float) -> float:
    if score <= 0:
        return 0.0
    if module == "recognition":
        max_score = RECOGNITION_MAX_SCORES.get(activity_id, 100)
        if score > max_score:
            return round(min(score, 100), 1)
        return round(min(score / max_score * 100, 100), 1)
    if module == "synthesis":
        max_score = SYNTHESIS_MAX_SCORES.get(activity_id, 10)
        return round(min(score / max_score * 100, 100), 1)
    return round(min(score, 100), 1)


def _latest_scores(db: Session, user_id: int, module: str) -> Dict[str, Submission]:
    subs = (
        db.query(Submission)
        .filter(Submission.user_id == user_id, Submission.module == module)
        .order_by(Submission.created_at.desc())
        .all()
    )
    latest: Dict[str, Submission] = {}
    for s in subs:
        if s.activity_id not in latest:
            latest[s.activity_id] = s
    return latest


def _cnn_scores(db: Session, user_id: int) -> Dict[str, float]:
    race = (
        db.query(CnnRaceRecord)
        .filter(CnnRaceRecord.user_id == user_id)
        .order_by(CnnRaceRecord.accuracy.desc())
        .first()
    )
    dataset = (
        db.query(CnnDatasetRecord)
        .filter(CnnDatasetRecord.user_id == user_id)
        .order_by(CnnDatasetRecord.created_at.desc())
        .first()
    )
    design = (
        db.query(CnnModelDesignRecord)
        .filter(CnnModelDesignRecord.user_id == user_id)
        .order_by(CnnModelDesignRecord.created_at.desc())
        .first()
    )
    upload = (
        db.query(CnnModelUploadRecord)
        .filter(CnnModelUploadRecord.user_id == user_id)
        .order_by(CnnModelUploadRecord.final_val_accuracy.desc())
        .first()
    )

    race_score = round(min((race.accuracy if race else 0), 100), 1)

    class_count = (dataset.class_count if dataset else 0) or 0
    dataset_score = round(
        (30 if class_count >= 10 else 22 if class_count >= 5 else 15 if dataset else 0)
        / 30 * 100,
        1,
    )

    conv_layers = (design.conv_layers if design else 0) or 0
    design_score = round(
        (30 if conv_layers >= 4 else 22 if conv_layers >= 2 else 15 if design else 0)
        / 30 * 100,
        1,
    )

    test_score = round(
        min((upload.final_val_accuracy or 0) * 100 if upload else 0, 100),
        1,
    )

    return {
        "race": race_score,
        "dataset": dataset_score,
        "model_design": design_score,
        "model_test": test_score,
    }


@router.get("/modules", response_model=List[ModuleInfo])
def list_modules(db: Session = Depends(get_db)):
    return db.query(Module).order_by(Module.id).all()


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(
    class_name: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    registered = db.query(Module).order_by(Module.id).all()

    if current_user.role in {"teacher", "admin"}:
        q = db.query(User).filter(User.role == "student")
        if current_user.role == "teacher":
            q = q.filter(User.class_name == current_user.class_name)
        if class_name:
            q = q.filter(User.class_name == class_name)
        if keyword:
            like = f"%{keyword}%"
            q = q.filter(
                (User.display_name.ilike(like)) | (User.username.ilike(like))
            )
        students = q.all()
        student_ids = [s.id for s in students]

        # Bulk fetch all submissions for these students (recognition + synthesis)
        subs = []
        if student_ids:
            subs = (
                db.query(Submission)
                .filter(Submission.user_id.in_(student_ids))
                .order_by(Submission.user_id, Submission.module, Submission.activity_id, Submission.created_at.desc())
                .all()
            )
        # Group: latest[user_id][module][activity_id] = submission
        latest: Dict[int, Dict[str, Dict[str, Submission]]] = {}
        for s in subs:
            latest.setdefault(s.user_id, {}).setdefault(s.module, {})
            if s.activity_id not in latest[s.user_id][s.module]:
                latest[s.user_id][s.module][s.activity_id] = s

        # Bulk fetch CNN records
        cnn_race, cnn_dataset, cnn_design, cnn_upload = {}, {}, {}, {}
        if student_ids:
            for row in db.query(CnnRaceRecord).filter(CnnRaceRecord.user_id.in_(student_ids)).all():
                cur = cnn_race.get(row.user_id)
                if cur is None or row.accuracy > cur.accuracy:
                    cnn_race[row.user_id] = row
            for row in db.query(CnnDatasetRecord).filter(CnnDatasetRecord.user_id.in_(student_ids)).order_by(CnnDatasetRecord.created_at.desc()).all():
                cnn_dataset.setdefault(row.user_id, row)
            for row in db.query(CnnModelDesignRecord).filter(CnnModelDesignRecord.user_id.in_(student_ids)).order_by(CnnModelDesignRecord.created_at.desc()).all():
                cnn_design.setdefault(row.user_id, row)
            for row in db.query(CnnModelUploadRecord).filter(CnnModelUploadRecord.user_id.in_(student_ids)).all():
                cur = cnn_upload.get(row.user_id)
                if cur is None or (row.final_val_accuracy or 0) > (cur.final_val_accuracy or 0):
                    cnn_upload[row.user_id] = row

        students_summary: Optional[List[StudentSummary]] = []
        for stu in students:
            rec = latest.get(stu.id, {}).get("recognition", {})
            syn = latest.get(stu.id, {}).get("synthesis", {})
            race = cnn_race.get(stu.id)
            dataset = cnn_dataset.get(stu.id)
            design = cnn_design.get(stu.id)
            upload = cnn_upload.get(stu.id)

            race_score = round(min((race.accuracy if race else 0), 100), 1)
            class_count = (dataset.class_count if dataset else 0) or 0
            dataset_score = round(
                (30 if class_count >= 10 else 22 if class_count >= 5 else 15 if dataset else 0) / 30 * 100, 1)
            conv_layers = (design.conv_layers if design else 0) or 0
            design_score = round(
                (30 if conv_layers >= 4 else 22 if conv_layers >= 2 else 15 if design else 0) / 30 * 100, 1)
            test_score = round(min((upload.final_val_accuracy or 0) * 100 if upload else 0, 100), 1)

            rec_avg = sum(_to_percent("recognition", act, s.score) for act, s in rec.items()) / len(rec) if rec else 0
            syn_avg = sum(_to_percent("synthesis", act, s.score) for act, s in syn.items()) / len(syn) if syn else 0
            cnn_avg = (race_score + dataset_score + design_score + test_score) / 4

            students_summary.append(StudentSummary(
                user_id=stu.id,
                username=stu.display_name or stu.username,
                class_name=stu.class_name,
                modules={
                    "recognition": round(rec_avg, 1),
                    "synthesis": round(syn_avg, 1),
                    "cnn": round(cnn_avg, 1),
                },
                plain_password=stu.plain_password if current_user.role == "admin" else None,
            ))
    else:
        students_summary = None

    uid = current_user.id
    rec_latest = _latest_scores(db, uid, "recognition")
    syn_latest = _latest_scores(db, uid, "synthesis")
    cnn_latest = _cnn_scores(db, uid)

    rec_activities = [
        ActivityScore(
            activity_id=act,
            score=_to_percent("recognition", act, rec_latest[act].score) if act in rec_latest else 0,
            created_at=rec_latest[act].created_at.isoformat() if act in rec_latest else None,
        )
        for act in RECOGNITION_ACTS
    ]
    syn_activities = [
        ActivityScore(
            activity_id=act,
            score=_to_percent("synthesis", act, syn_latest[act].score) if act in syn_latest else 0,
            created_at=syn_latest[act].created_at.isoformat() if act in syn_latest else None,
        )
        for act in SYNTHESIS_ACTS
    ]
    cnn_activities = [
        ActivityScore(activity_id=act, score=cnn_latest.get(act, 0))
        for act in CNN_ACTS
    ]

    def _avg(acts: List[ActivityScore]) -> float:
        scored = [a.score for a in acts if a.score > 0]
        return sum(scored) / len(scored) if scored else 0

    radar_values = [
        rec_activities[0].score,
        rec_activities[1].score,
        rec_activities[2].score,
        syn_activities[0].score,
        syn_activities[1].score,
        syn_activities[2].score,
        syn_activities[3].score,
        cnn_latest["race"],
        cnn_latest["dataset"],
        cnn_latest["model_design"],
        cnn_latest["model_test"],
    ]

    return DashboardResponse(
        user={
            "id": current_user.id,
            "username": current_user.username,
            "display_name": current_user.display_name or current_user.username,
            "role": current_user.role,
            "class_name": current_user.class_name,
            "must_change_password": current_user.must_change_password,
        },
        modules=[
            ModuleSummary(module_key="recognition", module_name="语音识别技术探究",
                          activities=rec_activities, avg_score=round(_avg(rec_activities), 1)),
            ModuleSummary(module_key="synthesis", module_name="语音合成技术探究",
                          activities=syn_activities, avg_score=round(_avg(syn_activities), 1)),
            ModuleSummary(module_key="cnn", module_name="卷积神经网络原理与应用",
                          activities=cnn_activities, avg_score=round(_avg(cnn_activities), 1)),
        ],
        radar={
            "dimensions": RECOGNITION_DIMS + SYNTHESIS_DIMS + CNN_DIMS,
            "values": [round(v, 1) for v in radar_values],
        },
        registered_modules=[ModuleInfo.model_validate(m) for m in registered],
        students_summary=students_summary,
    )
