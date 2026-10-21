from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import wave
from email.utils import formatdate
from typing import Any, Literal
from urllib.parse import urlencode

import aiohttp
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import Submission, User


router = APIRouter(prefix="/api/recognition", tags=["recognition"])

MAX_ATTEMPTS = 2
ACTIVITY_MAX_SCORES = {
    "dictation": 100.0,
    "tech_exploration": 6.0,
    "concept_sorting": 6.0,
}
ATTEMPT_META_KEYS = {"attemptCount", "attemptScores", "bestScore"}


class RecognitionAttemptCreate(BaseModel):
    activity_id: Literal["dictation", "tech_exploration", "concept_sorting"]
    score: float = 0
    raw_data: dict[str, Any] = Field(default_factory=dict)


def _parse_raw_data(submission: Submission | None) -> dict[str, Any]:
    if not submission:
        return {}
    try:
        value = json.loads(submission.raw_data or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _without_attempt_metadata(raw_data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw_data.items() if key not in ATTEMPT_META_KEYS}


@router.post("/attempts")
def save_recognition_attempt(
    body: RecognitionAttemptCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Save one recognition-module attempt and retain the higher score."""
    existing = (
        db.query(Submission)
        .filter(
            Submission.user_id == current_user.id,
            Submission.module == "recognition",
            Submission.activity_id == body.activity_id,
        )
        .first()
    )
    existing_raw = _parse_raw_data(existing)
    existing_attempts = int(existing_raw.get("attemptCount", 1 if existing else 0) or 0)
    if existing_attempts >= MAX_ATTEMPTS:
        raise HTTPException(status_code=409, detail="本活动最多只能提交两次")

    max_score = ACTIVITY_MAX_SCORES[body.activity_id]
    current_score = min(max(float(body.score), 0.0), max_score)
    previous_best = float(existing.score) if existing else -1.0
    saved_as_best = existing is None or current_score >= previous_best
    best_score = current_score if saved_as_best else previous_best
    attempt_count = existing_attempts + 1

    old_scores = existing_raw.get("attemptScores", [])
    attempt_scores = [float(value) for value in old_scores if isinstance(value, (int, float))]
    # Older records predate attempt metadata. Preserve their score as attempt one.
    if existing and not attempt_scores and existing_attempts:
        attempt_scores = [previous_best]
    attempt_scores.append(current_score)
    attempt_scores = attempt_scores[-MAX_ATTEMPTS:]

    if saved_as_best:
        best_raw = _without_attempt_metadata(body.raw_data)
    else:
        best_raw = _without_attempt_metadata(existing_raw)
    best_raw.update(
        {
            "attemptCount": attempt_count,
            "attemptScores": attempt_scores,
            "bestScore": best_score,
        }
    )

    if existing:
        existing.score = best_score
        existing.raw_data = json.dumps(best_raw, ensure_ascii=False)
    else:
        db.add(
            Submission(
                user_id=current_user.id,
                module="recognition",
                activity_id=body.activity_id,
                score=best_score,
                raw_data=json.dumps(best_raw, ensure_ascii=False),
            )
        )
    db.commit()
    return {
        "ok": True,
        "attemptCount": attempt_count,
        "attemptsRemaining": MAX_ATTEMPTS - attempt_count,
        "currentScore": current_score,
        "bestScore": best_score,
        "savedAsBest": saved_as_best,
    }


def _xfyun_credentials() -> tuple[str, str, str]:
    return (
        os.getenv("XFYUN_APP_ID", "").strip(),
        os.getenv("XFYUN_API_KEY", "").strip(),
        os.getenv("XFYUN_API_SECRET", "").strip(),
    )


@router.get("/xfyun/status")
def xfyun_status(_current_user: User = Depends(get_current_user)):
    return {"configured": all(_xfyun_credentials())}


def _build_xfyun_url(api_key: str, api_secret: str) -> str:
    host = "iat-api.xfyun.cn"
    path = "/v2/iat"
    date = formatdate(timeval=None, localtime=False, usegmt=True)
    signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(api_secret.encode("utf-8"), signature_origin.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
    return f"wss://{host}{path}?{urlencode({'authorization': authorization, 'date': date, 'host': host})}"


def _read_pcm_wav(content: bytes) -> bytes:
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="录音文件过大，请控制在60秒以内")
    try:
        with wave.open(io.BytesIO(content), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() != 16000:
                raise HTTPException(status_code=400, detail="录音必须为16kHz、16位、单声道 WAV")
            if wav_file.getnframes() > 16000 * 60:
                raise HTTPException(status_code=400, detail="单次录音不能超过60秒")
            return wav_file.readframes(wav_file.getnframes())
    except HTTPException:
        raise
    except (wave.Error, EOFError) as exc:
        raise HTTPException(status_code=400, detail="无法读取录音，请重新录制") from exc


async def _receive_xfyun(ws: aiohttp.ClientWebSocketResponse) -> str:
    parts: dict[int, str] = {}
    async for message in ws:
        if message.type == aiohttp.WSMsgType.TEXT:
            payload = json.loads(message.data)
            code = int(payload.get("code", 0))
            if code != 0:
                raise HTTPException(
                    status_code=502,
                    detail=f"讯飞识别失败：{payload.get('message', '未知错误')}（{code}）",
                )
            data = payload.get("data") or {}
            result = data.get("result") or {}
            words = "".join(
                candidate.get("w", "")
                for word_segment in result.get("ws", [])
                for candidate in word_segment.get("cw", [])[:1]
            )
            if words:
                parts[int(result.get("sn", len(parts)))] = words
            if int(data.get("status", 0)) == 2:
                break
        elif message.type in {aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED}:
            break
    return "".join(parts[index] for index in sorted(parts)).strip()


@router.post("/xfyun/recognize")
async def recognize_with_xfyun(
    language: Literal["zh_cn", "en_us"] = "zh_cn",
    audio: UploadFile = File(...),
    _current_user: User = Depends(get_current_user),
):
    app_id, api_key, api_secret = _xfyun_credentials()
    if not all((app_id, api_key, api_secret)):
        raise HTTPException(status_code=503, detail="讯飞语音识别尚未配置，请先填写服务端环境变量")

    pcm = _read_pcm_wav(await audio.read())
    if not pcm:
        return {"text": ""}

    business = {"language": language, "domain": "iat"}
    if language == "zh_cn":
        business["accent"] = "mandarin"

    timeout = aiohttp.ClientTimeout(total=75, connect=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(_build_xfyun_url(api_key, api_secret), heartbeat=20) as ws:
                receiver = asyncio.create_task(_receive_xfyun(ws))
                frame_size = 1280
                for offset in range(0, len(pcm), frame_size):
                    frame = pcm[offset : offset + frame_size]
                    payload: dict[str, Any] = {
                        "data": {
                            "status": 0 if offset == 0 else 1,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": base64.b64encode(frame).decode("utf-8"),
                        }
                    }
                    if offset == 0:
                        payload["common"] = {"app_id": app_id}
                        payload["business"] = business
                    await ws.send_json(payload)
                    await asyncio.sleep(0.04)
                await ws.send_json({"data": {"status": 2, "format": "audio/L16;rate=16000", "encoding": "raw", "audio": ""}})
                text = await receiver
                await ws.close(code=1000)
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="连接讯飞语音识别服务失败，请稍后重试") from exc

    return {"text": text}
