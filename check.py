"""Ask a local Ollama vision model about one visit."""
import json

ANSWERS = ("yes", "no", "unsure")
SCHEMA = {"type": "object", "properties": {
    "answer": {"type": "string", "enum": list(ANSWERS)}, "reason": {"type": "string"}},
    "required": ["answer", "reason"]}
RULES = ("Review these time-ordered crops of one shop visit. Answer only from visible evidence. "
         "Use unsure if the action or its outcome is unclear. Give a short visual reason. Return JSON.")


def ask(model, question, frames):
    import ollama

    try:
        reply = ollama.chat(model=model, messages=[
            {"role": "system", "content": RULES},
            {"role": "user", "content": question, "images": frames}],
            format=SCHEMA, think=False if model.startswith("qwen3.5:") else None,
            options={"temperature": 0, "num_ctx": 8192})
    except (ConnectionError, ollama.ResponseError) as exc:
        raise RuntimeError(f"Ollama could not run {model}. Check ollama serve and ollama pull {model}. {exc}") from exc
    try:
        result = json.loads(reply.message.content)
        if result["answer"] in ANSWERS and isinstance(result["reason"], str) and result["reason"].strip():
            return result["answer"], result["reason"].strip()
    except (ValueError, TypeError, KeyError):
        pass
    raise ValueError("Ollama returned an invalid answer. Retry the check or select a compatible vision model.")
