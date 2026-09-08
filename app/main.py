"""CLI runner for the MedAI LangGraph agent."""

import uuid

from app.services.agent_service import resume_agent, run_agent


def _print_interrupt(payload: dict) -> dict:
    print("\n--- HITL REVIEW REQUIRED ---")
    for key, value in payload.items():
        print(f"  {key}: {value}")
    action = input("Action [approve/reject/edit/change_type] (default: approve): ").strip() or "approve"
    note = ""
    if action in ("edit", "change_type"):
        note = input("Note (for edit: text, for change_type: image type key): ").strip()
    return {"action": action, "note": note or None}


def run_cli():
    thread_id = str(uuid.uuid4())

    print("MedAI LangGraph Agent")
    print("Commands: 'exit' to quit, 'image:<path>' to analyze an image")
    print(f"Thread ID: {thread_id}\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        image_path = None
        if user_input.lower().startswith("image:"):
            _, _, image_path = user_input.partition(":")
            image_path = image_path.strip()
            user_input = f"Analyze this medical image: {image_path}"

        result = run_agent(user_input, thread_id=thread_id, image_path=image_path)

        while result.get("status") == "awaiting_approval":
            feedback = _print_interrupt(result["interrupt"])
            result = resume_agent(
                thread_id,
                action=feedback["action"],
                note=feedback.get("note"),
            )

        answer = result.get("response") or "No response generated."
        print(f"\nAssistant: {answer}\n")

        if result.get("detected_disease"):
            print(f"  Disease: {result['detected_disease']} (confidence: {result.get('confidence')})")
        if result.get("prescription_draft"):
            print(f"  Prescription draft available (risk: {result.get('risk_level')})")
        print()


if __name__ == "__main__":
    run_cli()
