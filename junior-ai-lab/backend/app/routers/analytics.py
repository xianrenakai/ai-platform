"""Analytics API — time-series scores and class distributions for portal charts."""
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import Submission, User
from app.models.cnn.models import (
    CnnDatasetRecord, CnnModelDesignRecord, CnnModelUploadRecord, CnnRaceRecord,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

ACTIVITY_LABELS: dict[str, str] = {
    "dictation": "听写挑战", "tech_exploration": "技术探究", "concept_sorting": "概念排序",
    "activity1": "活动一",  "activity2": "活动二",  "activity3": "活动三", "activity4": "活动四",
    "race": "算术比赛", "dataset": "创建数据集", "model_design": "模型设计", "model_test": "模型测试",
}

MODULE_ACTIVITIES: dict[str, list[str]] = {
    "recognition": ["dictation", "tech_exploration", "concept_sorting"],
    "synthesis":   ["activity1", "activity2", "activity3", "activity4"],
    "cnn":         ["race", "dataset", "model_design", "model_test"],
}

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
    """Normalize every activity score to 0-100 for all analytics endpoints."""
    if score <= 0:
        return 0.0
    if module == "recognition":
        max_score = RECOGNITION_MAX_SCORES.get(activity_id, 100)
        return round(min(score / max_score * 100, 100), 1)
    if module == "synthesis":
        max_score = SYNTHESIS_MAX_SCORES.get(activity_id, 10)
        return round(min(score / max_score * 100, 100), 1)
    return round(min(score, 100), 1)

LEARNING_RESOURCES: dict[str, dict] = {
    "dictation":        {"tips": ["多听多练，熟悉语音识别常见错误模式", "注意清晰发音，减少连读和吞音"],
                         "links": [{"title": "科大讯飞AI学习平台", "url": "https://www.xfyun.cn/doc/"},
                                   {"title": "语音识别技术综述", "url": "https://zhuanlan.zhihu.com/p/27168991"}]},
    "tech_exploration": {"tips": ["了解声学模型与语言模型的分工", "噪声环境下模型性能会显著下降"],
                         "links": [{"title": "ASR技术综述", "url": "https://zhuanlan.zhihu.com/p/27168991"},
                                   {"title": "Kaldi语音识别工具", "url": "https://kaldi-asr.org/doc/"}]},
    "concept_sorting":  {"tips": ["流程：声音数字化→特征提取→声学模型→语言模型→识别结果", "重点理解MFCC特征提取"],
                         "links": [{"title": "MFCC特征提取详解", "url": "https://zhuanlan.zhihu.com/p/26729882"},
                                   {"title": "Mozilla DeepSpeech", "url": "https://github.com/mozilla/DeepSpeech"}]},
    "activity1":        {"tips": ["标点符号决定TTS的停顿和语气", "逗号、句号、问号对应不同韵律"],
                         "links": [{"title": "中文TTS技术解析", "url": "https://zhuanlan.zhihu.com/p/314212763"},
                                   {"title": "Coursera语音合成课程", "url": "https://www.coursera.org/"}]},
    "activity2":        {"tips": ["多音字需结合上下文判断", "'行'在动词语境读xing2，名词读hang2"],
                         "links": [{"title": "汉语多音字词典", "url": "https://www.zdic.net/"},
                                   {"title": "TTS多音字处理研究", "url": "https://arxiv.org/search/?query=polyphone"}]},
    "activity3":        {"tips": ["音素标注需了解汉语拼音声韵母体系", "声调对意义区分至关重要"],
                         "links": [{"title": "汉语拼音标准教程", "url": "https://www.pinyinput.com/"},
                                   {"title": "Festival TTS系统", "url": "http://www.cstr.ed.ac.uk/projects/festival/"}]},
    "activity4":        {"tips": ["语速控制在0.8-1.2倍效果最佳", "适当增加停顿可提升自然度"],
                         "links": [{"title": "语音合成韵律研究", "url": "https://arxiv.org/abs/2106.15561"},
                                   {"title": "微软Azure TTS SSML文档", "url": "https://docs.microsoft.com/azure/cognitive-services/speech-service/speech-synthesis-markup"}]},
    "race":             {"tips": ["反复练习提高心算速度", "先做简单题再做难题，建立信心"],
                         "links": [{"title": "Khan Academy数学", "url": "https://www.khanacademy.org/math"},
                                   {"title": "3Blue1Brown线性代数", "url": "https://www.3blue1brown.com/topics/linear-algebra"}]},
    "dataset":          {"tips": ["类别越均衡，模型效果越好", "增加样本多样性（不同字体、笔画粗细）"],
                         "links": [{"title": "MNIST数据集介绍", "url": "http://yann.lecun.com/exdb/mnist/"},
                                   {"title": "数据增强技术详解", "url": "https://zhuanlan.zhihu.com/p/41679153"},
                                   {"title": "Kaggle手写数字竞赛", "url": "https://www.kaggle.com/c/digit-recognizer"}]},
    "model_design":     {"tips": ["卷积层影响特征抽取深度", "池化层减少参数量防止过拟合"],
                         "links": [{"title": "CNN Explainer可视化工具", "url": "https://poloclub.github.io/cnn-explainer/"},
                                   {"title": "斯坦福CS231n课程", "url": "https://cs231n.stanford.edu/"},
                                   {"title": "LeNet-5经典论文", "url": "http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf"}]},
    "model_test":       {"tips": ["验证集准确率才是真实能力体现", "测试前确保图片预处理与训练时一致"],
                         "links": [{"title": "模型评估指标详解", "url": "https://scikit-learn.org/stable/modules/model_evaluation.html"},
                                   {"title": "TensorFlow Keras教程", "url": "https://www.tensorflow.org/tutorials"},
                                   {"title": "混淆矩阵解读", "url": "https://zhuanlan.zhihu.com/p/36397853"}]},
}


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
        result[s.activity_id].append({"date": _fmt(s.created_at), "score": _to_percent(module, s.activity_id, s.score)})
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
            latest[s.activity_id] = _to_percent(module, s.activity_id, s.score)
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
        score = _to_percent(module, activity_id, sub.score) if sub else 0
    return _ai_evaluate(activity_id, score, current_user.username)


@router.get("/classes")
def list_classes(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="teachers only")
    if current_user.role == "teacher":
        return {"classes": [current_user.class_name] if current_user.class_name else []}
    rows = db.query(User.class_name).filter(User.role == "student", User.class_name.isnot(None)).distinct().all()
    classes = sorted([r[0] for r in rows if r[0]])
    return {"classes": classes}


@router.get("/class/overview")
def class_overview(
    class_name: str = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="teachers only")
    q = db.query(User).filter(User.role == "student")
    if current_user.role == "teacher":
        q = q.filter(User.class_name == current_user.class_name)
    if class_name:
        q = q.filter(User.class_name == class_name)
    students = q.all()
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
def list_students(
    class_name: str = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="teachers only")
    q = db.query(User).filter(User.role == "student")
    if current_user.role == "teacher":
        q = q.filter(User.class_name == current_user.class_name)
    if class_name:
        q = q.filter(User.class_name == class_name)
    students = q.order_by(User.username).all()
    result = []
    for s in students:
        rec = _voice_latest(db, s.id, "recognition")
        syn = _voice_latest(db, s.id, "synthesis")
        cnn = _cnn_latest(db, s.id)
        result.append({"id": s.id, "username": s.display_name or s.username, "class_name": s.class_name,
                        "avg_recognition": _avg(rec), "avg_synthesis": _avg(syn), "avg_cnn": _avg(cnn),
                        "overall": round((_avg(rec) + _avg(syn) + _avg(cnn)) / 3, 1)})
    result.sort(key=lambda x: -x["overall"])
    return result


@router.get("/class/student/{student_id}")
def student_detail(student_id: int, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="teachers only")
    student = db.query(User).filter(User.id == student_id, User.role == "student").first()
    if not student:
        raise HTTPException(status_code=404, detail="student not found")
    if current_user.role == "teacher" and student.class_name != current_user.class_name:
        raise HTTPException(status_code=403, detail="教师只能查看自己班级的学生")
    rec_tl = _voice_timeline(db, student.id, "recognition")
    syn_tl = _voice_timeline(db, student.id, "synthesis")
    cnn_tl = _cnn_timeline(db, student.id)
    return {
        "student_id": student.id, "username": student.display_name or student.username, "class_name": student.class_name,
        "timelines": {**{f"recognition.{k}": v for k, v in rec_tl.items()},
                      **{f"synthesis.{k}": v for k, v in syn_tl.items()},
                      **{f"cnn.{k}": v for k, v in cnn_tl.items()}},
        "latest_scores": {**{f"recognition.{k}": v for k, v in _voice_latest(db, student.id, "recognition").items()},
                          **{f"synthesis.{k}": v for k, v in _voice_latest(db, student.id, "synthesis").items()},
                          **{f"cnn.{k}": v for k, v in _cnn_latest(db, student.id).items()}},
        "activity_labels": ACTIVITY_LABELS, "module_activities": MODULE_ACTIVITIES,
    }
