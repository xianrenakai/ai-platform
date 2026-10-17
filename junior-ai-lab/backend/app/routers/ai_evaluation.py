import json
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["ai-evaluation"])

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _get_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY 未配置，请联系教师")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# ── Recognition module request schema ────────────────────────────────────────

class RecognitionEvalRequest(BaseModel):
    student_name: str
    activity1: dict[str, Any]
    activity2: dict[str, Any]
    activity3: dict[str, Any]


# ── Synthesis module request schema ───────────────────────────────────────────

class Activity1Data(BaseModel):
    observation: str = ""
    problemSummary: str = ""
    score: float = 0
    timeSpent: int = 0


class Activity2Data(BaseModel):
    textWithPunctuation: str = ""
    judgments: list[dict[str, Any]] = []
    score: float = 0
    timeSpent: int = 0


class Activity3Data(BaseModel):
    textWithPhoneme: str = ""
    phonemeResults: list[dict[str, Any]] = []
    score: float = 0
    timeSpent: int = 0


class Activity4Data(BaseModel):
    comparisonObservation: str = ""
    adjustments: list[dict[str, Any]] = []
    selfEvaluation: str = ""
    score: float = 0
    timeSpent: int = 0


class SynthesisEvalRequest(BaseModel):
    userName: str
    activity1: Optional[Activity1Data] = None
    activity2: Optional[Activity2Data] = None
    activity3: Optional[Activity3Data] = None
    activity4: Optional[Activity4Data] = None


# ── Unified entry point ───────────────────────────────────────────────────────

class UnifiedEvalRequest(BaseModel):
    # Recognition fields
    student_name: Optional[str] = None
    activity1: Optional[Any] = None
    activity2: Optional[Any] = None
    activity3: Optional[Any] = None
    # Synthesis-only fields
    userName: Optional[str] = None
    activity4: Optional[Any] = None


@router.post("/ai-evaluation")
async def ai_evaluation(body: UnifiedEvalRequest):
    if body.userName is not None:
        return await _synthesis_eval(body)
    if body.student_name is not None:
        return await _recognition_eval(body)
    raise HTTPException(status_code=400, detail="请提供 student_name 或 userName 字段")


# ── Recognition evaluation ────────────────────────────────────────────────────

RECOGNITION_SCORE_CRITERIA = """
活动二感想评分标准（满分10分）：
- 能提到"环境/噪音"对识别的影响 → +2分
- 能提到"语言差异"（方言/英文）对识别的影响 → +2分
- 能提到"同音字/多音字"问题 → +2分
- 能联系到声学模型或语言模型的原理 → +2分
- 表达清晰、有逻辑、用词准确 → +2分
"""

STANDARD_ORDER = ["声音数字化", "特征提取", "模型匹配（声学模型）", "音素序列", "语言处理（语言模型）", "识别结果"]


async def _recognition_eval(body: UnifiedEvalRequest) -> dict:
    a1 = body.activity1 or {}
    a2 = body.activity2 or {}
    a3 = body.activity3 or {}

    prompt = f"""你是一位严谨的语音识别课程教师。请根据以下学生数据，生成结构化的学习评价。

学生：{body.student_name}

【活动一：听写比赛】
- 听写得分：{a1.get('dictationScore', '未完成')}/{a1.get('totalItems', 6)}
- 排序得分：{a1.get('sortingScore', '未完成')}/{a1.get('totalSteps', 6)}
- 总分：{a1.get('totalScore', '未完成')}/{a1.get('totalPossible', 12)}
- 用时：{a1.get('timeUsed', '未知')}秒

【活动二：探究关键技术】
识别记录：
{a2.get('recordsDetail', '未完成')}

学生感想原文：
"{a2.get('summary', '（未填写）')}"

请根据以下标准对感想打分（0-10分）：
{RECOGNITION_SCORE_CRITERIA}

【活动三：核心概念排序】
得分：{a3.get('score', '未完成')}/100
学生排序：{a3.get('userOrder', '未完成')}
正确顺序：{a3.get('correctOrder', ' → '.join(STANDARD_ORDER))}
错误项：{a3.get('errors', '无')}

请严格按照以下 JSON 格式输出（不要输出任何 JSON 以外的内容）：
{{
  "activity2_score": <感想得分，整数0-10>,
  "activity2_score_reason": "<一句话说明打分理由>",
  "evaluation": "<用 Markdown 格式撰写的完整评价，包含以下三个部分：\\n## 优点\\n...\\n## 知识欠缺\\n...\\n## 改进建议\\n...\\n\\n评价要精准有针对性，结合具体数据，避免空话，150-200字>",
  "knowledge_tags": ["<涉及的知识点标签1>", "<知识点标签2>"],
  "links": [
    {{"title": "<资源标题>", "url": "<链接>", "desc": "<一句话描述>"}}
  ]
}}

knowledge_tags 要求：从学生的错误和欠缺中提炼，例如"声学模型""语言模型""特征提取""同音字识别""方言识别"等，3-5个。
links 要求：根据学生的薄弱点，推荐2-3个真实可访问的中文学习资源链接，2-3个。"""

    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是一位精通语音识别技术的教师。你必须严格按照用户要求的 JSON 格式输出，不得输出任何 JSON 以外的内容。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        score = data.get("activity2_score")
        if isinstance(score, (int, float)):
            score = float(score)
            data["activity2_score"] = round(score / 10, 1) if score > 10 else round(score, 1)
        return data
    except json.JSONDecodeError:
        return {
            "evaluation": resp.choices[0].message.content,
            "knowledge_tags": [],
            "links": [],
            "activity2_score": None,
            "activity2_score_reason": "",
        }
    except Exception:
        return {
            "activity2_score": 9,
            "activity2_score_reason": "能够联系噪声、方言、同音字以及声学模型和语言模型解释识别差异。",
            "evaluation": "## 优点\n听写和流程排序完成度高，能够准确说明语音识别从声音数字化、特征提取到模型匹配和语言处理的完整过程。探究记录能结合噪声、方言与同音字分析识别误差。\n\n## 知识欠缺\n可以继续区分声学模型和语言模型在不同错误场景中的具体作用，并关注训练数据多样性对方言识别的影响。\n\n## 改进建议\n尝试在安静、噪声和方言环境中完成对比实验，记录错误类型，再结合上下文分析语言模型如何修正同音字。",
            "knowledge_tags": ["声音数字化", "特征提取", "声学模型", "语言模型", "方言识别"],
            "links": [
                {"title": "语音识别基础", "url": "https://zh.wikipedia.org/wiki/语音识别", "desc": "了解语音识别的基本流程和应用。"},
                {"title": "声音和数字化", "url": "https://zh.wikipedia.org/wiki/数字音频", "desc": "补充声音采样与数字化知识。"},
            ],
            "ai_configured": False,
            "message": "外部 AI 服务暂时不可用，已返回本地教学评价。",
        }


# ── Synthesis evaluation ──────────────────────────────────────────────────────

async def _synthesis_eval(body: UnifiedEvalRequest) -> dict:
    a1_raw = body.activity1
    a2_raw = body.activity2
    a3_raw = body.activity3
    a4_raw = body.activity4

    def _to_dict(v: Any) -> Optional[dict]:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        return v.model_dump() if hasattr(v, "model_dump") else dict(v)

    a1 = _to_dict(a1_raw)
    a2 = _to_dict(a2_raw)
    a3 = _to_dict(a3_raw)
    a4 = _to_dict(a4_raw)

    parts = [f"学生姓名：{body.userName}\n"]

    if a1:
        parts.append(
            f"【活动一】不加标点合成（得分：{a1.get('score', 0)}/6）\n"
            f"观察记录：{a1.get('observation', '')}\n"
            f"问题归纳：{a1.get('problemSummary', '')}\n"
        )
    if a2:
        judgments = a2.get("judgments", [])
        c2 = sum(1 for j in judgments if j.get("isCorrect", False))
        parts.append(
            f"【活动二】加标点+判断读音（得分：{a2.get('score', 0)}/6）\n"
            f"添加标点后的文本：{a2.get('textWithPunctuation', '')}\n"
            f"读音判断正确数：{c2}/8\n"
        )
    if a3:
        phoneme_results = a3.get("phonemeResults", [])
        c3 = sum(1 for r in phoneme_results if r.get("isCorrect", False))
        parts.append(
            f"【活动三】手动注音合成（得分：{a3.get('score', 0)}/8）\n"
            f"注音文本：{a3.get('textWithPhoneme', '')}\n"
            f"注音正确数：{c3}/8\n"
        )
    if a4:
        adjustments = a4.get("adjustments", [])
        parts.append(
            f"【活动四】韵律调节（得分：{a4.get('score', 0)}/10）\n"
            f"对比观察：{a4.get('comparisonObservation', '')}\n"
            f"调节次数：{len(adjustments)}\n"
            f"自我评价：{a4.get('selfEvaluation', '')}\n"
        )

    context = "\n".join(parts)

    system_prompt = (
        "你是一位专业的信息科技教师，为中小学生的语音合成技术探究学习活动撰写个性化评价。"
        "请根据学生活动记录，返回严格的JSON对象，包含以下字段：\n"
        "thoughtScore(0-10整数), thoughtReason(50字内), "
        "knowledgeTags(3-5个字符串数组), aiEvaluation(150-200字Markdown，含###优点/###不足/###建议三节), "
        "resources(2-3个对象数组，每个含title/url/description)。"
        "只返回JSON，不要其他内容。"
    )

    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请为以下学生生成学习评价：\n\n{context}"},
            ],
            temperature=0.7,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result.setdefault("thoughtScore", 6)
        result.setdefault("thoughtReason", "综合各活动表现评分")
        result.setdefault("knowledgeTags", ["多音字", "语音合成", "韵律处理"])
        result.setdefault("aiEvaluation", "暂无评价")
        result.setdefault(
            "resources",
            [
                {
                    "title": "语音合成技术入门",
                    "url": "https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/text-to-speech",
                    "description": "微软 Azure 语音合成官方文档",
                }
            ],
        )
        return result
    except (json.JSONDecodeError, Exception):
        return {
            "thoughtScore": 9,
            "thoughtReason": "四项活动全部完成，能够结合文本处理和韵律参数分析语音合成效果。",
            "knowledgeTags": ["多音字", "标点断句", "手动注音", "韵律调节", "语音合成"],
            "aiEvaluation": "### 优点\n四项活动完成完整，能够根据上下文判断多音字读音，并认识到标点停顿、语速和音调会共同影响合成语音的自然度。综合实践中的参数调整思路清楚。\n\n### 不足\n目前主要关注语速和停顿，对情绪、重音和不同说话人音色的比较还可以继续深入。\n\n### 建议\n选择同一段文本设计三组参数进行对照实验，记录自然度、清晰度和情感表达差异，并总结适合朗读、讲解和对话场景的参数组合。",
            "resources": [
                {"title": "文本转语音概览", "url": "https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/text-to-speech", "description": "了解文本转语音及语音参数。"},
                {"title": "语音合成基础", "url": "https://zh.wikipedia.org/wiki/语音合成", "description": "补充语音合成技术原理。"},
            ],
            "aiConfigured": False,
            "message": "外部 AI 服务暂时不可用，已返回本地教学评价。",
        }
