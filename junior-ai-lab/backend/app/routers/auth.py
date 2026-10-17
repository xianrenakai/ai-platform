from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import os
from threading import Lock
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import ACCESS_TOKEN_EXPIRE_HOURS, SESSION_COOKIE_NAME, create_access_token, get_current_user, hash_password, verify_password
from app.database import get_db
from app.models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])
LOGIN_WINDOW = timedelta(minutes=15)
MAX_LOGIN_FAILURES = 5
_login_failures: dict[str, deque[datetime]] = defaultdict(deque)
_login_lock = Lock()


def _failure_key(request: Request, username: str) -> str:
    client = request.client.host if request.client else "unknown"
    return f"{client}:{username.casefold()}"


def _prune_failures(key: str, now: datetime) -> deque[datetime]:
    failures = _login_failures[key]
    cutoff = now - LOGIN_WINDOW
    while failures and failures[0] < cutoff:
        failures.popleft()
    return failures


def _cookie_secure(request: Request) -> bool:
    """Use Secure cookies behind HTTPS while keeping classroom HTTP usable."""
    configured = os.getenv("COOKIE_SECURE", "auto").strip().lower()
    if configured in {"1", "true", "yes", "on"}:
        return True
    if configured in {"0", "false", "no", "off"}:
        return False
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    public_host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    public_host = public_host.split(",", 1)[0].split(":", 1)[0].strip().lower()
    public_https_host = (
        public_host.endswith(".trycloudflare.com")
        or public_host.endswith(".lhr.life")
        or public_host.endswith(".pinggy.link")
        or public_host.endswith(".pinggy-free.link")
    )
    return forwarded_proto == "https" or request.url.scheme == "https" or public_https_host


class LoginRequest(BaseModel):
    username: str = Field(..., description="姓名")
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    display_name: Optional[str] = None
    role: str
    class_name: Optional[str] = None
    must_change_password: bool = False

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    user: UserOut


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    name = body.username.strip()
    key = _failure_key(request, name)
    now = datetime.now(timezone.utc)
    with _login_lock:
        failures = _prune_failures(key, now)
        if len(failures) >= MAX_LOGIN_FAILURES:
            retry_after = max(1, int((failures[0] + LOGIN_WINDOW - now).total_seconds()))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"登录失败次数过多，请在{max(1, (retry_after + 59) // 60)}分钟后重试",
                headers={"Retry-After": str(retry_after)},
            )
    user = db.query(User).filter(User.username == name).first()
    if not user or not user.is_active or not verify_password(body.password, user.password_hash):
        with _login_lock:
            _prune_failures(key, now).append(now)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="姓名或密码错误，请检查后重试",
        )
    token = create_access_token(user.id, user.username, user.role)
    with _login_lock:
        _login_failures.pop(key, None)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        path="/",
    )
    return TokenResponse(
        user=UserOut.model_validate(user),
    )


@router.post("/logout")
def logout(request: Request, response: Response):
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        path="/",
    )
    return {"ok": True}


@router.post("/register")
def register_disabled():
    raise HTTPException(status_code=403, detail="学校平台已关闭公共注册，请联系本班教师创建账号")


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)


@router.post("/change-password")
def change_password(
    body: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(body.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="当前密码不正确")
    if body.current_password == body.new_password:
        raise HTTPException(status_code=400, detail="新密码不能与当前密码相同")
    current_user.password_hash = hash_password(body.new_password)
    current_user.must_change_password = False
    db.commit()
    return {"ok": True}
