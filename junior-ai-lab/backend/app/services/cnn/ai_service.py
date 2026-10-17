import json
import os
import re
from typing import Any, Dict, List

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

DEFAULT_REPORT = {
    "activity_analysis": {
        "race": "已开始进行算术比赛练习，可继续提升速度与稳定性。",
        "dataset": "数据集创建体现了实践意识，建议继续增加样本多样性。",
        "model_design": "模型设计已有基础，可进一步理解卷积层与全连接层作用。",
        "model_test": "模型测试能帮助验证效果，建议多对比不同模型表现。",
    },
    "comprehensive_review": "你已经完成了重要的CNN学习实践，具备从数据、建模到测试的完整体验。继续保持动手探索，会取得更明显进步。",
    "strengths": ["学习流程意识较完整", "具备实践操作能力", "能够关注模型效果"],
    "weaknesses": ["可增加数据样本多样性", "可深入理解模型结构", "可加强结果复盘分析"],
    "learning_plan": ["补充未完成的学习活动", "优化数据集质量", "尝试不同CNN结构", "记录并对比测试结果"],
    "career_advice": "建议继续学习CNN经典结构、模型调优和图像识别项目实践。",
}


def _api_key() -> str:
    return (os.getenv("DEEPSEEK_API_KEY") or "").strip()


def _is_configured() -> bool:
    k = _api_key()
    return bool(k) and k not in {"", "你的API_KEY"}


def _client():
    if not _OPENAI_AVAILABLE:
        return None
    return OpenAI(api_key=_api_key(), base_url="https://api.deepseek.com", timeout=30.0)


def _extract_json(content: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    return DEFAULT_REPORT.copy()


def generate_evaluation_report(student_data: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_configured() or not _OPENAI_AVAILABLE:
        return {**DEFAULT_REPORT, "ai_configured": False, "message": "AI 功能需要配置 DeepSeek API Key 后使用。"}

    prompt = f"""你是一位资深教育评估专家。请根据以下学生数据生成学习评价报告，严格按JSON格式返回：
{{
  "activity_analysis": {{"race":"...","dataset":"...","model_design":"...","model_test":"..."}},
  "comprehensive_review": "综合评语（100字以内）",
  "strengths": ["优势1","优势2","优势3"],
  "weaknesses": ["待改进1","待改进2","待改进3"],
  "learning_plan": ["建议1","建议2","建议3","建议4"],
  "career_advice": "进阶学习方向（80字以内）"
}}
学生数据：{json.dumps(student_data, ensure_ascii=False, default=str)}"""

    try:
        resp = _client().chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        report = _extract_json(resp.choices[0].message.content or "")
        report["ai_configured"] = True
        return report
    except Exception:
        return {**DEFAULT_REPORT, "ai_configured": True, "message": "AI 服务暂时不可用，已返回默认学习评价。"}


def chat_with_ai(student_data: Dict[str, Any], message: str, history: List[dict]) -> Dict[str, Any]:
    if not _is_configured() or not _OPENAI_AVAILABLE:
        return {"reply": "AI 功能需要配置 DeepSeek API Key 后使用。", "ai_configured": False}

    msg = (message or "").strip() or "请根据我的学习数据给出一句个性化欢迎语。"
    system = f"""你是CNN学习平台的AI助教，引导式教学，每次回答100字以内，多鼓励学生。
学生数据：{json.dumps(student_data, ensure_ascii=False, default=str)}"""

    messages = [{"role": "system", "content": system}]
    for h in history[-10:]:
        if h.get("role") in {"user", "assistant"} and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": msg})

    try:
        resp = _client().chat.completions.create(model="deepseek-chat", messages=messages, temperature=0.8, max_tokens=500)
        return {"reply": resp.choices[0].message.content or "继续加油！", "ai_configured": True}
    except Exception:
        return {"reply": "继续加油！你已经完成了不少学习活动。", "ai_configured": True}
