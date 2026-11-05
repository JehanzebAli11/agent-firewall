"""Training-data provenance rule: training labels/records may come only from code with exact ground truth
or from the local Gemma teacher on the Spark. Outputs of Claude / OpenAI / Gemini / Jev (or any LLMJudge)
may be evaluated but never trained on."""
ALLOWED_TRAINING_SOURCES = {"code", "teacher:gemma-4-31b"}
FORBIDDEN_MARKERS = ("claude", "anthropic", "openai", "gpt", "gemini", "jev")


def check_trainable(records) -> None:
    for r in records:
        src = r.get("label_source", "code")
        if src not in ALLOWED_TRAINING_SOURCES or any(m in src.lower() for m in FORBIDDEN_MARKERS):
            raise ValueError(f"record {r.get('id')} has non-trainable label_source={src!r}")
