from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Password hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── Scheme definitions ────────────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_access_token(
    data: dict,
    settings: Settings,
    expires_delta: Optional[timedelta] = None,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str, settings: Settings) -> dict:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Dependency: require a valid caller (API key OR Bearer JWT)
# ─────────────────────────────────────────────────────────────────────────────

async def require_auth(
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Returns caller identity string.  Accepts either:
      - X-API-Key header matching one of the configured static keys
      - Authorization: Bearer <JWT> issued by /auth/token
    """
    # 1. Static API key
    if api_key:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        valid_hashes = {
            hashlib.sha256(k.encode()).hexdigest()
            for k in settings.api_keys_list
        }
        if key_hash in valid_hashes:
            logger.debug("auth.api_key_accepted")
            return f"apikey:{api_key[:8]}***"

    # 2. JWT Bearer
    if bearer:
        payload = decode_access_token(bearer.credentials, settings)
        subject = payload.get("sub")
        if subject:
            logger.debug("auth.jwt_accepted", sub=subject)
            return subject

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required: provide X-API-Key or Bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: generate a new random API key (use in admin tooling)
# ─────────────────────────────────────────────────────────────────────────────

def generate_api_key() -> str:
    return f"ltx_{secrets.token_urlsafe(32)}"
