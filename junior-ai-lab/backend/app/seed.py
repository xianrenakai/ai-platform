import json
import os
import random
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.auth import hash_password
from app.database import Base, SessionLocal, engine
from app.models import Module, Submission, User
from app.models.cnn.models import (
    CnnDatasetRecord, CnnModelDesignRecord, CnnModelUploadRecord,
    CnnRaceRecord, CnnRecognitionRecord,
)

TEACHERS = [
    ("史秦恺", "teacher123", "七年级1班"),
    ("孙丹", "teacher123", "七年级2班"),
    ("陈敏", "teacher123", "七年级1班"),
    ("赵志远", "teacher123", "七年级2班"),
]

STUDENTS_CLASS1 = [
    "王哲", "李娜", "张浩宇", "刘雨桐", "陈子轩",
    "赵梦瑶", "周子涵", "吴子豪", "郑欣怡", "林俊杰",
    "黄雅琪", "张一辰", "刘若曦", "王梓安", "李沐晴",
    "赵俊哲", "陈思睿", "孙逸飞", "周若琳", "吴宇恒",
    "郑乐怡", "林书瑶", "黄子骞", "张欣妍", "刘子昂",
    "陈雨萱", "赵子豪", "王若溪", "李铭泽", "周子瑜",
]

STUDENTS_CLASS2 = [
    "方晨曦", "谢佳宁", "唐子睿", "蒋欣瑶", "韩宇轩",
    "冯雨晴", "程乐天", "朱思远", "魏梦琪", "潘嘉豪",
    "叶子萱", "邓浩然", "许若彤", "宋子默", "傅雨欣",
    "汪俊逸", "马晓彤", "薛子涵", "贺雨桐", "苏铭远",
    "龚思琪", "柳浩宇", "邵欣悦", "卢子扬", "曹雨萌",
    "侯思远", "孟佳琳", "严子豪", "戴欣怡", "乔梦瑶",
]

MODULES = [
    {"module_key": "recognition", "module_name": "语音识别技术探究",
     "description": "体验听写、探究关键技术、概念排序", "icon": "🎤", "route_path": "http://localhost:5174"},
    {"module_key": "synthesis",   "module_name": "语音合成技术探究",
     "description": "多音字注音、标点断句、韵律调节",    "icon": "🔊", "route_path": "http://localhost:5175"},
    {"module_key": "cnn",         "module_name": "卷积神经网络原理与应用",
     "description": "算术比赛、创建数据集、模型设计与测试","icon": "🧠", "route_path": "http://localhost:5176"},
]

PRODUCTION_MODULE_ROUTES = {
    "recognition": "/recognition/",
    "synthesis": "/synthesis/",
    "cnn": "/cnn/",
}
DEVELOPMENT_MODULE_ROUTES = {
    "recognition": "http://localhost:5174",
    "synthesis": "http://localhost:5175",
    "cnn": "http://localhost:5176",
}

RECOGNITION_ACTIVITIES = ["dictation", "tech_exploration", "concept_sorting"]
SYNTHESIS_ACTIVITIES   = ["activity1", "activity2", "activity3", "activity4"]

RECOGNITION_RANGES = {"dictation": (55, 100), "tech_exploration": (50, 100), "concept_sorting": (50, 100)}
SYNTHESIS_RANGES   = {"activity1": (3, 6),    "activity2": (3, 6),  "activity3": (4, 8), "activity4": (5, 10)}

CNN_ARCHITECTURES = [
    (1, [32], 32, 1, 128, "relu", 10),
    (2, [32, 64], 96, 2, 128, "relu", 10),
    (2, [64, 128], 192, 2, 256, "relu", 10),
    (3, [32, 64, 128], 224, 3, 256, "relu", 10),
    (3, [64, 128, 256], 448, 3, 512, "relu", 10),
    (4, [32, 64, 64, 128], 288, 4, 256, "relu", 10),
    (4, [64, 128, 128, 256], 576, 4, 512, "relu", 10),
]
CNN_MODEL_NAMES = [
    "LeNet风格", "简单CNN", "双层卷积", "三层卷积",
    "深层卷积", "宽层CNN", "标准CNN", "轻量CNN",
    "增强CNN", "精简版LeNet", "自定义模型", "实验版CNN",
]


def _dt(days_back: int = 28, hours_var: int = 23) -> datetime:
    return datetime.utcnow() - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, hours_var),
        minutes=random.randint(0, 59),
    )


# ── CNN: multiple attempts per student for rich time-series data ─────────────

def _seed_cnn_for_student(db: Session, student: User) -> None:
    """
    Each student gets 3-5 race attempts, 2-3 dataset records, 2-3 model designs,
    2-3 model uploads, and recognition records — spread over 28 days so the
    line charts show meaningful progression.
    """
    # Profile: each student has a personal "ability" tier that shapes their scores
    ability = random.uniform(0.45, 1.0)   # 0.45=weak .. 1.0=excellent

    # ── Race: 3-5 attempts, scores improve over time ──────────────────────────
    n_race = random.randint(3, 5)
    # 基础正确率：45%-75%（更真实的范围）
    base_accuracy_rate = ability * random.uniform(0.45, 0.75)
    for i in range(n_race):
        improvement = i * random.uniform(0.02, 0.08)  # 每次提升2%-8%
        total_q = random.choice([10, 15, 20])
        
        # 计算本次正确率（0-1范围），加入随机波动，确保不超过95%
        accuracy_rate = min(0.95, base_accuracy_rate + improvement + random.uniform(-0.08, 0.08))
        accuracy_rate = max(0.35, accuracy_rate)  # 确保最低35%
        
        # 根据正确率计算正确题数
        correct = int(accuracy_rate * total_q)  # 使用int而不是round，避免四舍五入到总数
        wrong = total_q - correct
        
        # 重新计算实际正确率
        actual_accuracy = correct / total_q if total_q > 0 else 0.0
        
        db.add(CnnRaceRecord(
            user_id=student.id,
            total_questions=total_q,
            correct_count=correct,
            wrong_count=wrong,
            accuracy=actual_accuracy,  # 0-1范围
            score=round(actual_accuracy * 100, 1),  # 得分 = 正确率 × 100
            created_at=_dt(days_back=28 - i * 5),
        ))

    # ── Dataset: 2-3 records ─────────────────────────────────────────────────
    n_dataset = random.randint(2, 3)
    for i in range(n_dataset):
        sel_digits = list(range(random.randint(3, 10)))
        sel_counts = {str(d): random.randint(50, 150) for d in sel_digits}
        total_imgs = sum(sel_counts.values())
        ts = _dt(days_back=25 - i * 7).strftime("%Y%m%d_%H%M%S")
        db.add(CnnDatasetRecord(
            user_id=student.id,
            dataset_name=f"{student.username}_{ts}",
            digits=json.dumps(sel_digits),
            counts=json.dumps(sel_counts),
            total_images=total_imgs,
            class_count=len(sel_digits),
            created_at=_dt(days_back=25 - i * 7),
        ))

    # ── Model design: 2-3 attempts ───────────────────────────────────────────
    n_design = random.randint(2, 3)
    for i in range(n_design):
        arch = CNN_ARCHITECTURES[min(i, len(CNN_ARCHITECTURES) - 1)]
        conv_l, conv_f, total_cf, pool_l, fc_n, fc_act, out_n = arch
        model_name = random.choice(CNN_MODEL_NAMES)
        db.add(CnnModelDesignRecord(
            user_id=student.id,
            model_name=model_name,
            conv_layers=conv_l,
            conv_filters=json.dumps(conv_f),
            total_conv_filters=total_cf,
            pool_layers=pool_l,
            fc_neurons=fc_n,
            fc_activation=fc_act,
            output_neurons=out_n,
            structure_summary=f"输入→{'→'.join([f'Conv({f})→Pool' for f in conv_f])}→FC({fc_n})→输出({out_n})",
            created_at=_dt(days_back=20 - i * 6),
        ))

    # ── Model upload: 2-3 uploads, accuracy improves ─────────────────────────
    n_upload = random.randint(2, 3)
    last_upload = None
    for i in range(n_upload):
        sel_digits = list(range(random.randint(5, 10)))
        num_c = len(sel_digits)
        # First upload is weak, last is strong
        acc_base = ability * (0.65 + i * 0.10)
        final_val_acc = round(min(0.99, acc_base + random.uniform(-0.03, 0.05)), 3)
        mid = f"{num_c}_{_dt(days_back=15 - i * 5).strftime('%Y%m%d_%H%M%S')}_{student.id}_{i}"
        model_name = CNN_MODEL_NAMES[i % len(CNN_MODEL_NAMES)]
        upload = CnnModelUploadRecord(
            user_id=student.id,
            model_id=mid,
            model_filename=f"{model_name.replace(' ', '_')}.h5",
            model_path=f"uploads/{mid}.h5",
            final_accuracy=round(final_val_acc + 0.01, 3),
            final_val_accuracy=final_val_acc,
            test_accuracy=round(final_val_acc - 0.01, 3),
            epochs=random.choice([5, 10, 15, 20]),
            conv_layers=i + 1,
            pool_layers=i + 1,
            dense_layers=1,
            total_params=random.randint(50000, 500000),
            train_data_json=json.dumps({"epochs": 10, "num_classes": num_c}),
            class_mapping_json=json.dumps({str(j): str(d) for j, d in enumerate(sel_digits)}),
            created_at=_dt(days_back=15 - i * 5),
        )
        db.add(upload)
        db.flush()
        last_upload = upload

    # ── Recognition records tied to best upload ───────────────────────────────
    if last_upload:
        for _ in range(random.randint(2, 5)):
            sel_digits_for_rec = list(range(random.randint(5, 10)))
            db.add(CnnRecognitionRecord(
                user_id=student.id,
                model_upload_id=last_upload.id,
                predicted_class=str(random.choice(sel_digits_for_rec)),
                confidence=round(random.uniform(0.75, 0.99), 4),
                image_format="白底黑字",
                created_at=_dt(7),
            ))


# ── Voice seed helpers (unchanged) ───────────────────────────────────────────

def _rec_raw(activity_id: str, score: float) -> dict:
    if activity_id == "dictation":
        a, b = round(score * 0.5), round(score * 0.5)
        return {"dictationScore": a, "sortingScore": b, "totalScore": a + b,
                "timeUsed": random.randint(40, 120)}
    if activity_id == "tech_exploration":
        correct = round(score / 100 * 6)
        records = []
        for i, (text, lang) in enumerate(zip(
            ["机器人", "robot", "机器人", "智", "智能音箱", "葭沚老街"],
            ["普通话", "英语", "方言", "普通话", "普通话", "普通话"],
        )):
            ok = i < correct
            records.append({"text": text, "language": lang,
                             "recognized": text if ok else text + "（误）", "isAccurate": ok})
        return {"records": records,
                "summary": random.choice(["嘈杂环境下识别率明显下降。", "同音字对语言模型是最大挑战。"])}
    if activity_id == "concept_sorting":
        standard = ["声音数字化", "特征提取", "模型匹配（声学模型）", "音素序列", "语言处理（语言模型）", "识别结果"]
        correct_c = round(score / 100 * 6)
        order = standard[:]
        for _ in range(min(6 - correct_c, 3)):
            a, b = random.sample(range(len(order)), 2)
            order[a], order[b] = order[b], order[a]
        return {"userOrder": order, "score": correct_c, "timeUsed": random.randint(30, 90)}
    return {"timeSpent": random.randint(120, 600)}


def _syn_raw(activity_id: str, score: float) -> dict:
    if activity_id == "activity1":
        return {"observation": "不加标点时，语音合成的断句很奇怪。",
                "problemSummary": "标点符号是语音合成断句的重要依据。",
                "score": score, "timeSpent": random.randint(60, 300)}
    if activity_id == "activity2":
        correct_c = round(score / 6 * 8)
        judgments = [{"index": i, "isCorrect": i < correct_c} for i in range(8)]
        return {"judgments": judgments, "score": score, "timeSpent": random.randint(90, 360)}
    if activity_id == "activity3":
        correct_c = round(score / 8 * 8)
        phoneme_results = [{"index": i, "isCorrect": i < correct_c} for i in range(8)]
        return {"phonemeResults": phoneme_results, "score": score, "timeSpent": random.randint(120, 480)}
    if activity_id == "activity4":
        return {"comparisonObservation": "调整语速后朗读更流畅。",
                "score": score, "timeSpent": random.randint(180, 720)}
    return {"timeSpent": random.randint(180, 720)}


# ── Main seed ────────────────────────────────────────────────────────────────

def seed_database(db: Session) -> None:
    existing_admin = db.query(User).filter(User.username == "admin").first()
    if not existing_admin:
        db.add(User(
            username="admin", display_name="管理员",
            password_hash=hash_password("admin123"), role="admin",
            class_name=None, must_change_password=True, is_active=True,
        ))
        db.commit()

    routes = PRODUCTION_MODULE_ROUTES if os.getenv("APP_ENV", "development").lower() == "production" else DEVELOPMENT_MODULE_ROUTES
    for module_key, route_path in routes.items():
        module = db.query(Module).filter(Module.module_key == module_key).first()
        if module:
            module.route_path = route_path
    db.commit()

    if db.query(User).filter(User.role != "admin").count() > 0:
        return

    for name, password, class_name in TEACHERS:
        db.add(User(username=name, display_name=name, password_hash=hash_password(password),
                    role="teacher", class_name=class_name))
    for name in STUDENTS_CLASS1:
        db.add(User(username=name, display_name=name, password_hash=hash_password("123456"),
                    role="student", class_name="七年级1班"))
    for name in STUDENTS_CLASS2:
        db.add(User(username=name, display_name=name, password_hash=hash_password("123456"),
                    role="student", class_name="七年级2班"))
    db.flush()

    for mod in MODULES:
        db.add(Module(**mod))
    db.flush()

    students = db.query(User).filter(User.role == "student").all()
    now = datetime.utcnow()

    for student in students:
        # Voice recognition — one submission per activity
        for act in RECOGNITION_ACTIVITIES:
            lo, hi = RECOGNITION_RANGES[act]
            score = round(random.uniform(lo, hi), 1)
            db.add(Submission(
                user_id=student.id, module="recognition", activity_id=act, score=score,
                raw_data=json.dumps(_rec_raw(act, score), ensure_ascii=False),
                created_at=_dt(),
            ))
        # Voice synthesis — one submission per activity
        for act in SYNTHESIS_ACTIVITIES:
            lo, hi = SYNTHESIS_RANGES[act]
            score = round(random.uniform(lo, hi), 1)
            db.add(Submission(
                user_id=student.id, module="synthesis", activity_id=act, score=score,
                raw_data=json.dumps(_syn_raw(act, score), ensure_ascii=False),
                created_at=_dt(),
            ))
        # CNN — multiple attempts
        _seed_cnn_for_student(db, student)

    db.commit()


def run_seed() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()
