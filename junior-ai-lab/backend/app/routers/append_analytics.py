import sys
path = r'd:\Users\Administrator\Desktop\AI-Platform\junior-ai-lab\backend\app\routers\analytics.py'

append_content = '''

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(dt) -> str:
    return dt.strftime("%m/%d") if dt else ""


def _cnn_timeline(db, uid: int) -> dict:
    races   = db.query(CnnRaceRecord).filter(CnnRaceRecord.user_id == uid).order_by(CnnRaceRecord.created_at).all()
    dsets   = db.query(CnnDatasetRecord).filter(CnnDatasetRecord.user_id == uid).order_by(CnnDatasetRecord.created_at).all()
    designs = db.query(CnnModelDesignRecord).filter(CnnModelDesignRecord.user_id == uid).order_by(CnnModelDesignRecord.created_at).all()
    uploads = db.query(CnnModelUploadRecord).filter(CnnModelUploadRecord.user_id == uid).order_by(CnnModelUploadRecord.created_at).all()
    return {
        "race":         [{"date": _fmt(r.created_at), "score": round(min(r.accuracy, 100), 1)} for r in races],
        "dataset":      [{"date": _fmt(d.created_at), "score": round((30 if d.class_count >= 10 else 22 if d.class_count >= 5 else 15) / 30 * 100, 1)} for d in dsets],
        "model_design": [{"date": _fmt(d.created_at), "score": round((30 if d.conv_layers >= 4 else 22 if d.conv_layers >= 2 else 15) / 30 * 100, 1)} for d in designs],
        "model_test":   [{"date": _fmt(u.created_at), "score": round(min((u.final_val_accuracy or 0) * 100, 100), 1)} for u in uploads],
    }


def _voice_timeline(db, uid: int, module: str) -> dict:
    from collections import defaultdict
    subs = db.query(Submission).filter(Submission.user_id == uid, Submission.module == module).order_by(Submission.created_at).all()
    result = defaultdict(list)
    for s in subs:
        result[s.activity_id].append({"date": _fmt(s.created_at), "score": round(s.score, 1)})
    return dict(result)


def _cnn_latest(db, uid: int) -> dict:
    race   = db.query(CnnRaceRecord).filter(CnnRaceRecord.user_id == uid).order_by(CnnRaceRecord.accuracy.desc()).first()
    ds     = db.query(CnnDatasetRecord).filter(CnnDatasetRecord.user_id == uid).order_by(CnnDatasetRecord.created_at.desc()).first()
    design = db.query(CnnModelDesignRecord).filter(CnnModelDesignRecord.user_id == uid).order_by(CnnModelDesignRecord.created_at.desc()).first()
    upload = db.query(CnnModelUploadRecord).filter(CnnModelUploadRecord.user_id == uid).order_by(CnnModelUploadRecord.final_val_accuracy.desc()).first()
    return {
        "race":         round(min((race.accuracy if race else 0), 100), 1),
        "dataset":      round((30 if (ds.class_count if ds else 0) >= 10 else 22 if (ds.class_count if ds else 0) >= 5 else 15 if ds else 0) / 30 * 100, 1),
        "model_design": round((30 if (design.conv_layers if design else 0) >= 4 else 22 if (design.conv_layers if design else 0) >= 2 else 15 if design else 0) / 30 * 100, 1),
        "model_test":   round(min((upload.final_val_accuracy or 0) * 100 if upload else 0, 100), 1),
    }


def _voice_latest(db, uid: int, module: str) -> dict:
    subs = db.query(Submission).filter(Submission.user_id == uid, Submission.module == module).order_by(Submission.created_at.desc()).all()
    latest: dict = {}
    for s in subs:
        if s.activity_id not in latest:
            latest[s.activity_id] = round(s.score, 1)
    return latest


def _avg(d: dict) -> float:
    vals = [v for v in d.values() if v > 0]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _ai_evaluate(activity_id: str, score: float, username: str) -> dict:
    label = ACTIVITY_LABELS.get(activity_id, activity_id)
    res   = LEARNING_RESOURCES.get(activity_id, {"tips": [], "links": []})
    if score >= 90:   level, body = "A+", f"你在[{label}]中表现非常出色！得分 {score} 分，已掌握核心知识点。"
    elif score >= 75: level, body = "A",  f"你在[{label}]中表现良好，得分 {score} 分，还有小幅提升空间。"
    elif score >= 60: level, body = "B",  f"你在[{label}]中得分 {score} 分，基础已掌握，建议加强练习。"
    else:             level, body = "C",  f"你在[{label}]中得分 {score} 分，建议认真回顾知识点后重新尝试。"
    return {"activity_id": activity_id, "activity_label": label, "score": score,
            "level": level, "comment": f"{username}同学，{body}", "tips": res["tips"], "resources": res["links"]}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/student/timeline")
def student_timeline(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    uid = current_user.id
    rec_tl = _voice_timeline(db, uid, "recognition")
    syn_tl = _voice_timeline(db, uid, "synthesis")
    cnn_tl = _cnn_timeline(db, uid)
    return {
        "username": current_user.username,
        "timelines": {**{f"recognition.{k}": v for k, v in rec_tl.items()},
                      **{f"synthesis.{k}": v for k, v in syn_tl.items()},
                      **{f"cnn.{k}": v for k, v in cnn_tl.items()}},
        "latest_scores": {**{f"recognition.{k}": v for k, v in _voice_latest(db, uid, "recognition").items()},
                          **{f"synthesis.{k}": v for k, v in _voice_latest(db, uid, "synthesis").items()},
                          **{f"cnn.{k}": v for k, v in _cnn_latest(db, uid).items()}},
        "activity_labels": ACTIVITY_LABELS,
        "module_activities": MODULE_ACTIVITIES,
    }


@router.get("/student/evaluate")
def evaluate_activity(activity_id: str = Query(...), db: Session = Depends(get_db),
                      current_user: User = Depends(get_current_user)):
    module = next((m for m, acts in MODULE_ACTIVITIES.items() if activity_id in acts), None)
    if not module:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}")
    if module == "cnn":
        score = _cnn_latest(db, current_user.id).get(activity_id, 0)
    else:
        sub = (db.query(Submission)
                 .filter(Submission.user_id == current_user.id, Submission.module == module,
                         Submission.activity_id == activity_id)
                 .order_by(Submission.created_at.desc()).first())
        score = round(sub.score, 1) if sub else 0
    return _ai_evaluate(activity_id, score, current_user.username)


@router.get("/class/overview")
def class_overview(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role != "teacher":
        raise HTTPException(status_code=403, detail="teachers only")
    students = db.query(User).filter(User.role == "student").all()
    all_scores: dict = {a: [] for a in ACTIVITY_LABELS}
    for stu in students:
        rec = _voice_latest(db, stu.id, "recognition")
        syn = _voice_latest(db, stu.id, "synthesis")
        cnn = _cnn_latest(db, stu.id)
        for act in MODULE_ACTIVITIES["recognition"]: all_scores[act].append(rec.get(act, 0))
        for act in MODULE_ACTIVITIES["synthesis"]:   all_scores[act].append(syn.get(act, 0))
        for act in MODULE_ACTIVITIES["cnn"]:         all_scores[act].append(cnn.get(act, 0))

    def _buckets(scores: list) -> dict:
        b = {"0-60": 0, "60-75": 0, "75-90": 0, "90-100": 0}
        for s in scores:
            if s < 60:   b["0-60"] += 1
            elif s < 75: b["60-75"] += 1
            elif s < 90: b["75-90"] += 1
            else:        b["90-100"] += 1
        valid = [s for s in scores if s > 0]
        return {"distribution": b,
                "avg": round(sum(scores) / len(scores), 1) if scores else 0,
                "max": round(max(valid), 1) if valid else 0,
                "min": round(min(valid), 1) if valid else 0}

    return {"total_students": len(students),
            "activities": {a: {**_buckets(all_scores[a]), "label": ACTIVITY_LABELS[a]} for a in ACTIVITY_LABELS},
            "module_activities": MODULE_ACTIVITIES, "activity_labels": ACTIVITY_LABELS}


@router.get("/class/students")
def list_students(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role != "teacher":
        raise HTTPException(status_code=403, detail="teachers only")
    students = db.query(User).filter(User.role == "student").order_by(User.username).all()
    result = []
    for s in students:
        rec = _voice_latest(db, s.id, "recognition")
        syn = _voice_latest(db, s.id, "synthesis")
        cnn = _cnn_latest(db, s.id)
        result.append({"id": s.id, "username": s.username, "class_name": s.class_name,
                        "avg_recognition": _avg(rec), "avg_synthesis": _avg(syn), "avg_cnn": _avg(cnn),
                        "overall": round((_avg(rec) + _avg(syn) + _avg(cnn)) / 3, 1)})
    result.sort(key=lambda x: -x["overall"])
    return result


@router.get("/class/student/{student_id}")
def student_detail(student_id: int, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    if current_user.role != "teacher":
        raise HTTPException(status_code=403, detail="teachers only")
    student = db.query(User).filter(User.id == student_id, User.role == "student").first()
    if not student:
        raise HTTPException(status_code=404, detail="student not found")
    rec_tl = _voice_timeline(db, student.id, "recognition")
    syn_tl = _voice_timeline(db, student.id, "synthesis")
    cnn_tl = _cnn_timeline(db, student.id)
    return {
        "student_id": student.id, "username": student.username, "class_name": student.class_name,
        "timelines": {**{f"recognition.{k}": v for k, v in rec_tl.items()},
                      **{f"synthesis.{k}": v for k, v in syn_tl.items()},
                      **{f"cnn.{k}": v for k, v in cnn_tl.items()}},
        "latest_scores": {**{f"recognition.{k}": v for k, v in _voice_latest(db, student.id, "recognition").items()},
                          **{f"synthesis.{k}": v for k, v in _voice_latest(db, student.id, "synthesis").items()},
                          **{f"cnn.{k}": v for k, v in _cnn_latest(db, student.id).items()}},
        "activity_labels": ACTIVITY_LABELS, "module_activities": MODULE_ACTIVITIES,
    }
'''

with open(path, 'a', encoding='utf-8') as f:
    f.write(append_content)
print("appended OK, total lines:", len(open(path, encoding='utf-8').read().splitlines()))
