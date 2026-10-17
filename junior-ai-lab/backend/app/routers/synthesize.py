from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services.tts_service import synthesize_speech

router = APIRouter(prefix="/api", tags=["synthesize"])

SHENXIANJU_TEXT = (
    "台州神仙居，国家5A级仙境，融山水之灵，汇自然之奇。"
    "峰峦叠翠，云雾缭绕，仿若仙人居所，一步一景，令人心旷神怡，流连忘返。"
)


class SynthesizeRequest(BaseModel):
    text: str = ""
    ssml: Optional[str] = None
    rate: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=1.0, ge=0.5, le=2.0)
    volume: float = Field(default=100.0, ge=0, le=200)
    voice: Optional[str] = None
    pauseDuration: int = Field(default=0, ge=0, le=1000)
    usePunctuationPause: bool = False


@router.post("/synthesize")
async def synthesize(body: SynthesizeRequest):
    content = (body.text or "").strip()
    if not content and not (body.ssml and body.ssml.strip()):
        raise HTTPException(status_code=400, detail="请提供要合成的文本")

    try:
        audio = await synthesize_speech(
            text=content,
            ssml=body.ssml,
            rate=body.rate,
            pitch=body.pitch,
            volume=body.volume,
            voice=body.voice or "zh-CN-XiaoxiaoNeural",
            pause_duration=body.pauseDuration if body.usePunctuationPause else 0,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"语音合成失败：{exc}",
        ) from exc

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )


@router.get("/generate-machine-audio")
async def generate_machine_audio():
    try:
        audio = await synthesize_speech(
            text=SHENXIANJU_TEXT,
            voice="zh-CN-XiaoxiaoNeural",
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"机器音频生成失败：{exc}") from exc

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=machine-voice.mp3"},
    )
