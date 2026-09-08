"""MedAI Streamlit web interface."""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from app.config import HF_TOKEN, UPLOAD_DIR
from app.models.image_models import IMAGE_MODELS
from app.services.agent_service import resume_agent, run_agent
from app.storage.patient_store import list_patients

st.set_page_config(
    page_title="MedAI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a5276; margin-bottom: 0.2rem; }
    .sub-header { color: #566573; margin-bottom: 1.5rem; }
    .hitl-box {
        border: 2px solid #f39c12;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        background: #fef9e7;
        margin: 1rem 0;
    }
    .result-box {
        border-left: 4px solid #1a5276;
        padding: 0.8rem 1rem;
        background: #eaf2f8;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .disclaimer { font-size: 0.85rem; color: #7f8c8d; font-style: italic; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _init_session() -> None:
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "messages": [],
        "pending_interrupt": None,
        "last_result": {},
        "uploaded_image_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _new_chat() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending_interrupt = None
    st.session_state.last_result = {}
    st.session_state.uploaded_image_path = None


def _save_upload(uploaded_file) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".png"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(uploaded_file.getvalue())
    return str(dest)


def _append_assistant(content: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": content})


def _handle_result(result: dict) -> None:
    st.session_state.last_result = result
    if result.get("status") == "awaiting_approval":
        st.session_state.pending_interrupt = result.get("interrupt")
    else:
        st.session_state.pending_interrupt = None
        response = result.get("response") or "No response generated."
        _append_assistant(response)


def _process_message(message: str, *, image_path: str | None = None, image_type: str | None = None) -> None:
    st.session_state.messages.append({"role": "user", "content": message})
    with st.spinner("Analyzing..."):
        result = run_agent(
            message,
            thread_id=st.session_state.thread_id,
            image_path=image_path,
            image_type=image_type,
        )
    _handle_result(result)


def _render_hitl_panel() -> None:
    interrupt = st.session_state.pending_interrupt
    if not interrupt:
        return

    with st.container(border=True):
        st.warning("Human review required before continuing")
        st.markdown(f"**{interrupt.get('message', 'Review required')}**")
        st.json({k: v for k, v in interrupt.items() if k not in ("message", "options")})

        itype = interrupt.get("type", "")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            approve = st.button("Approve", type="primary", use_container_width=True, key="hitl_approve")
        with col2:
            reject = st.button("Reject", use_container_width=True, key="hitl_reject")
        with col3:
            edit = st.button("Edit", use_container_width=True, key="hitl_edit")
        with col4:
            change_type = (
                st.button("Change type", use_container_width=True, key="hitl_change")
                if itype == "image_confirmation"
                else False
            )

        note = st.text_area(
            "Notes / edits / image type key",
            placeholder="For edit: revised prescription text. For change type: e.g. chest_xray, brain_mri",
            key="hitl_note",
        )

        if approve:
            _resume_hitl("approve", note or None)
        elif reject:
            _resume_hitl("reject", note or None)
        elif edit:
            _resume_hitl("edit", note or None)
        elif change_type:
            _resume_hitl("change_type", note or None)


def _resume_hitl(action: str, note: str | None) -> None:
    if action in ("edit", "change_type") and not note:
        st.error("Please provide a note for edit or change type.")
        return

    with st.spinner("Processing review..."):
        result = resume_agent(
            st.session_state.thread_id,
            action=action,
            note=note,
        )
    _handle_result(result)
    st.rerun()


def _render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    _render_hitl_panel()

    if not st.session_state.pending_interrupt:
        prompt = st.chat_input("Ask about symptoms, diseases, treatments, or patient info...")
        if prompt:
            _process_message(prompt)
            st.rerun()


def _render_results_panel() -> None:
    result = st.session_state.last_result
    if not result:
        st.info("Results will appear here after analysis.")
        return

    if result.get("detected_disease"):
        st.markdown(
            f'<div class="result-box"><b>Detected:</b> {result["detected_disease"]}'
            f'<br><b>Confidence:</b> {result.get("confidence", "N/A")}</div>',
            unsafe_allow_html=True,
        )

    if result.get("risk_level"):
        risk = result["risk_level"]
        color = {"low": "green", "medium": "orange", "high": "red"}.get(risk, "gray")
        st.markdown(f"**Risk level:** :{color}[{risk.upper()}]")

    if result.get("prescription_draft"):
        with st.expander("Prescription draft", expanded=True):
            st.markdown(result["prescription_draft"])

    partial = result.get("partial_state") or {}
    if partial.get("patient_info"):
        with st.expander("Patient info"):
            st.json(partial["patient_info"])


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Settings")

        if st.button("New conversation", use_container_width=True):
            _new_chat()
            st.rerun()

        st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")

        if not HF_TOKEN:
            st.error("HF_TOKEN not set in .env")

        st.divider()
        st.markdown("### Medical image")

        uploaded = st.file_uploader(
            "Upload scan (X-ray, MRI, skin, retina)",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
        )

        image_type = st.selectbox(
            "Image type (optional override)",
            options=["auto"] + list(IMAGE_MODELS.keys()),
            format_func=lambda x: "Auto-detect" if x == "auto" else x.replace("_", " ").title(),
        )

        image_message = st.text_input(
            "Message with image",
            value="Analyze this medical image",
        )

        if st.button("Analyze image", use_container_width=True, disabled=uploaded is None):
            path = _save_upload(uploaded)
            st.session_state.uploaded_image_path = path
            selected_type = None if image_type == "auto" else image_type
            _process_message(image_message, image_path=path, image_type=selected_type)
            st.rerun()

        if st.session_state.uploaded_image_path:
            st.image(st.session_state.uploaded_image_path, caption="Last uploaded image", use_container_width=True)

        st.divider()
        st.markdown("### Supported scans")
        for key in IMAGE_MODELS:
            st.caption(f"- {key.replace('_', ' ').title()}")


def _render_patients_tab() -> None:
    patients = list_patients(limit=100)
    if not patients:
        st.info("No saved patient records yet. Save patient info via the chat (info intent + HITL approve).")
        return

    for p in patients:
        with st.expander(f"Patient #{p.get('id')} — {p.get('name') or 'Unnamed'}"):
            st.write(f"**Created:** {p.get('created_at', 'N/A')}")
            st.json({k: v for k, v in p.items() if k not in ("id", "created_at")})


def main() -> None:
    _init_session()
    _render_sidebar()

    st.markdown('<p class="main-header">MedAI Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Medical Q&A, image analysis, and clinician-reviewed prescriptions</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="disclaimer">Not a substitute for professional medical advice. '
        "High-risk outputs require clinician approval.</p>",
        unsafe_allow_html=True,
    )

    tab_chat, tab_patients = st.tabs(["Chat", "Patients"])

    with tab_chat:
        col_chat, col_results = st.columns([2, 1])
        with col_chat:
            _render_chat()
        with col_results:
            st.markdown("#### Analysis")
            _render_results_panel()

    with tab_patients:
        _render_patients_tab()


if __name__ == "__main__":
    main()
