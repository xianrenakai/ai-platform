import asyncio
import io
import re
from html import unescape
from typing import Optional

import edge_tts

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
TTS_TIMEOUT_SECONDS = 15


def _rate_str(rate: float) -> str:
    return f"{int(round((rate - 1.0) * 100)):+d}%"


def _pitch_str(pitch: float) -> str:
    return f"{int(round((pitch - 1.0) * 50)):+d}Hz"


def _volume_str(volume: float) -> str:
    return f"{int(round(volume - 100)):+d}%"


def _plain_text(text: str, ssml: Optional[str] = None, pause_duration: int = 0) -> str:
    """Convert activity input to the exact plain text edge-tts should speak."""
    source = (ssml or text or "").strip()
    # Activity three uses pronunciation annotations. Edge TTS creates its own
    # SSML envelope, so reliable homophones are used instead of nested markup.
    source = re.sub(r"行\[hang\]", "航", source, flags=re.IGNORECASE)
    source = re.sub(r"行\[xing\]", "形", source, flags=re.IGNORECASE)
    source = re.sub(r"<break\s+time=[\"'][^\"']+[\"']\s*/?>", "，", source, flags=re.IGNORECASE)
    source = re.sub(r"<[^>]+>", "", source)
    source = unescape(source)
    if pause_duration > 0:
        commas = "，" * max(1, min(3, round(pause_duration / 300)))
        source = re.sub(r"[，、；：]+", commas, source)
        source = re.sub(r"[。！？]+", "。" + commas, source)
    return source.strip()


async def synthesize_speech(
    text: str,
    ssml: Optional[str] = None,
    rate: float = 1.0,
    pitch: float = 1.0,
    volume: float = 100.0,
    voice: str = DEFAULT_VOICE,
    pause_duration: int = 0,
) -> bytes:
    spoken_text = _plain_text(text, ssml, pause_duration)
    if not spoken_text:
        raise ValueError("没有可合成的文字")

    communicate = edge_tts.Communicate(
        spoken_text,
        voice,
        rate=_rate_str(rate),
        pitch=_pitch_str(pitch),
        volume=_volume_str(volume),
    )
    buffer = io.BytesIO()

    async def _stream():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

    try:
        await asyncio.wait_for(_stream(), timeout=TTS_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("语音合成超时（超过15秒），请稍后重试") from exc

    data = buffer.getvalue()
    if not data:
        raise ValueError("语音合成失败，未生成音频数据")
    return data
