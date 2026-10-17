import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.auth import get_current_user
from app.database import get_db
from app.models import User

router = APIRouter(prefix="/api/cnn/materials", tags=["cnn-materials"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MATERIALS_DIR = BASE_DIR / "uploads" / "materials"
INDEX_FILE = MATERIALS_DIR / "materials.json"
DEFAULT_TEMPLATE_ID = "modeltrain-template"
ALLOWED_SUFFIXES = {".pdf", ".doc", ".docx", ".py", ".zip", ".ppt", ".pptx", ".txt"}


class MaterialResponse(BaseModel):
    id: str
    title: str
    filename: str
    fileType: str
    size: int
    uploader: str
    createdAt: str
    downloadUrl: str
    builtin: bool = False


def _student_template_path() -> Path:
    candidates = [
        BASE_DIR.parent / "student" / "modeltrain.py",
        BASE_DIR / "modeltrain.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="modeltrain.py 模板文件不存在")


def _ensure_store() -> None:
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text("[]", encoding="utf-8")


def _read_index() -> list[dict]:
    _ensure_store()
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _write_index(records: list[dict]) -> None:
    _ensure_store()
    INDEX_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _download_url(material_id: str) -> str:
    return f"/api/cnn/materials/{material_id}/download"


def _builtin_material() -> dict:
    path = _student_template_path()
    return {
        "id": DEFAULT_TEMPLATE_ID,
        "title": "CNN模型训练模板 modeltrain.py",
        "filename": "modeltrain.py",
        "fileType": ".py",
        "size": path.stat().st_size,
        "uploader": "平台内置",
        "createdAt": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "downloadUrl": _download_url(DEFAULT_TEMPLATE_ID),
        "builtin": True,
    }


def _serialize(record: dict) -> MaterialResponse:
    return MaterialResponse(
        id=str(record["id"]),
        title=str(record.get("title") or record.get("filename") or "学习资料"),
        filename=str(record.get("filename") or "material"),
        fileType=str(record.get("fileType") or Path(str(record.get("filename", ""))).suffix.lower()),
        size=int(record.get("size") or 0),
        uploader=str(record.get("uploader") or "教师"),
        createdAt=str(record.get("createdAt") or datetime.utcnow().isoformat()),
        downloadUrl=_download_url(str(record["id"])),
        builtin=bool(record.get("builtin", False)),
    )


@router.get("", response_model=list[MaterialResponse])
def list_materials(current_user: User = Depends(get_current_user)):
    records = [_builtin_material(), *_read_index()]
    return [_serialize(record) for record in records]


@router.post("/upload", response_model=MaterialResponse)
def upload_material(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"teacher", "admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="只有教师可以添加学习资料")

    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 PDF、Word、PPT、Python、ZIP、TXT 文件")

    material_id = uuid4().hex
    stored_name = f"{material_id}{suffix}"
    target = MATERIALS_DIR / stored_name
    _ensure_store()

    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    record = {
        "id": material_id,
        "title": Path(original_name).stem or "学习资料",
        "filename": original_name,
        "storedName": stored_name,
        "fileType": suffix,
        "size": target.stat().st_size,
        "uploader": current_user.username,
        "createdAt": datetime.utcnow().isoformat(),
        "builtin": False,
    }
    records = _read_index()
    records.insert(0, record)
    _write_index(records)
    return _serialize(record)


@router.get("/{material_id}/download")
def download_material(material_id: str, current_user: User = Depends(get_current_user)):
    if material_id == DEFAULT_TEMPLATE_ID:
        return FileResponse(
            _student_template_path(),
            media_type="text/x-python",
            filename="modeltrain.py",
        )

    record = next((item for item in _read_index() if str(item.get("id")) == material_id), None)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="资料不存在")

    stored_name = Path(str(record.get("storedName", ""))).name
    path = MATERIALS_DIR / stored_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="资料文件不存在")

    return FileResponse(path, filename=str(record.get("filename") or stored_name))
