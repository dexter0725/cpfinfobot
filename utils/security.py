"""Password utilities for protecting the Streamlit app."""
from __future__ import annotations

import hmac
import os
from typing import Optional

import streamlit as st


def _get_secret(key: str, env_var: str) -> Optional[str]:
    secret = st.secrets.get(key) if hasattr(st, "secrets") else None  # type: ignore[attr-defined]
    return secret or os.getenv(env_var)


def _get_password_secret() -> Optional[str]:
    return _get_secret("password", "CPF_APP_PASSWORD")


def _get_admin_secret() -> Optional[str]:
    return _get_secret("admin_password", "CPF_ADMIN_PASSWORD")


def check_password() -> bool:
    """Return True if the user enters the correct password stored in secrets/env."""

    required_password = _get_password_secret()
    if not required_password:
        st.error("App password is not configured. Set `password` in .streamlit/secrets.toml or `CPF_APP_PASSWORD`.")
        return False

    def password_entered() -> None:
        entered = st.session_state.get("password", "")
        if hmac.compare_digest(entered, required_password):
            st.session_state.password_correct = True
            st.session_state.pop("password", None)
        else:
            st.session_state.password_correct = False

    if st.session_state.get("password_correct"):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if st.session_state.get("password_correct") is False:
        st.error("😕 Password incorrect")
    return False


def admin_password_configured() -> bool:
    return _get_admin_secret() is not None


def verify_admin_password(value: str) -> bool:
    secret = _get_admin_secret()
    if not secret:
        return False
    return hmac.compare_digest(value, secret)


__all__ = ["check_password", "verify_admin_password", "admin_password_configured"]
