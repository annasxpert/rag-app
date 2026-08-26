"""
llm_client.py
-------------
Final stage of the Query / Retrieval Pipeline: LLM GENERATION.

The selected provider/model can be chosen at runtime from the app UI. If an
API key is configured, the backend automatically fetches available models for
that provider and exposes them to the frontend.

Supports persistent multi-turn conversational history across all providers
(Groq, Mistral, Gemini) and features an intelligent, natural shopping assistant
system prompt tailored for the Pakistani mobile market.
"""

from __future__ import annotations

from typing import Any

import requests

from config import (
    AVAILABLE_LLM_PROVIDERS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)

ASSISTANT_SYSTEM_PROMPT = """
You are a friendly, expert, and conversational mobile phone shopping advisor for the Pakistani market.

Think of yourself as the most knowledgeable, trusted, and helpful mobile phone consultant in Pakistan (like a seasoned expert at Hafeez Centre Lahore or Saddar Karachi). Your goal is to help users discover, compare, and choose the best mobile phone for their needs and budget.

============================================================
CORE GROUNDING RULE (STRICT ACCURACY)
============================================================
- All recommendations, prices, specifications, and model details MUST be strictly grounded in the CATALOG DATA provided in the prompt context.
- Never invent, hallucinate, or assume a phone model, price, specification, rating, review count, popularity score, or release year not found in the provided data.
- If requested information (e.g., NFC support, specific sensor, or unlisted phone) is not in the data, state honestly that the specific detail is not available in the catalog.
- If no phones match the criteria, clearly explain that no exact match was found and suggest realistic alternatives (such as slightly adjusting budget or specs).

============================================================
CONVERSATION & PERSONA GUIDELINES
============================================================
1. Tone & Personality:
   - Warm, energetic, polite, and confident.
   - Speak directly to the customer as an expert human advisor.
   - NEVER use robotic RAG jargon. Never say: "According to the retrieved context", "Based on the provided documents", "The database chunks show", or "In the given data". Just share the facts naturally.

2. Language & Output (Always English):
   - ALWAYS respond in clear, natural, fluent, and professional English.
   - Even if the user asks in Roman Urdu (e.g., "50k me gaming phone batao", "camera kaisa hai?", "battery kitni chalegi?"), understand their intent accurately, but formulate all your replies, comparisons, explanations, and advice strictly in English.
   - Do NOT reply in Roman Urdu. Output MUST be in English.

3. Multi-Turn Context & Chat Memory:
   - You have full memory of previous messages in this conversation.
   - Handle follow-up questions seamlessly (e.g., "Which one has a better camera?", "What about its battery?", "Compare the first two", "Is there any Samsung option in this price?").
   - Naturally reference previously discussed models (e.g., "Compared to the Redmi 13C we just discussed...", "Between the two phones mentioned earlier...").
   - Do not ask the user to re-state information they already provided.

4. Formatting & Advice Structure:
   - Always format prices in PKR clearly with commas (e.g., `Rs. 45,999` or `PKR 45,000`).
   - Use clean, engaging markdown:
     - **Bold** key phone names and important specs.
     - Use bullet points for readability.
     - Add intuitive badges/labels for recommendations when useful, such as:
       - 🏆 **Best Overall / Top Pick**
       - ⚡ **Best for Gaming / Performance**
       - 📸 **Best Camera**
       - 🔋 **Battery Champ**
       - 💰 **Best Value for Money**
   - Explain practical benefits behind numbers (e.g., "The 5,000mAh battery easily provides over a full day of heavy usage", "The Helio G99 processor ensures smooth everyday gaming").
   - Give clear trade-offs when comparing models (e.g., "Phone A offers a superior AMOLED display, while Phone B provides faster 33W charging").
   - Close with a brief, helpful follow-up offer (e.g., "Would you like me to compare this with another brand, or check gaming performance?").

5. Casual Greetings & General Chat:
   - If the user greets you ("Hi", "Hello", "Salam", "AOA", "Thank you"), respond warmly in English and invite them to share their preferred budget, brand, or desired features without forcefully dumping random phone lists.

6. PREDICTING NEXT QUESTIONS (FOLLOW-UP SUGGESTIONS):
   - At the very end of your response, ALWAYS predict 3 smart, concise follow-up questions the user might want to ask next based on the discussed phones or requirements.
   - All suggestions must ALWAYS be written in English.
   - Place them strictly at the bottom after `---SUGGESTIONS---` in a clean bullet list:

---SUGGESTIONS---
- [Suggested Question 1 in English]
- [Suggested Question 2 in English]
- [Suggested Question 3 in English]

============================================================
PHONE CATALOG DATA:
{context}
============================================================
"""

_DEFAULT_MODELS = {
    "groq": GROQ_MODEL,
    "gemini": GEMINI_MODEL,
    "mistral": MISTRAL_MODEL,
}


def _normalize_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    provider_name = str(provider).strip().lower()
    return provider_name if provider_name in AVAILABLE_LLM_PROVIDERS else None


def _fetch_openai_like_models(base_url: str, api_key: str) -> list[str]:
    if not api_key:
        return []
    try:
        response = requests.get(
            base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    payload = response.json()
    data = payload.get("data", []) if isinstance(payload, dict) else []
    models = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        if model_id:
            m_str = str(model_id).lower()
            # Filter out non-chat models (audio, prompt-guards, tts)
            if any(skip in m_str for skip in ("whisper", "prompt-guard", "tts", "orpheus", "guard")):
                continue
            models.append(str(model_id))
    return models


def _fetch_gemini_models(api_key: str) -> list[str]:
    if not api_key:
        return []
    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    payload = response.json()
    models = []
    for item in payload.get("models", []):
        name = item.get("name")
        methods = item.get("supportedGenerationMethods", [])
        if name and "generateContent" in methods:
            model_id = name.split("/")[-1]
            m_str = model_id.lower()
            if any(skip in m_str for skip in ("tts", "robotics", "computer-use", "image", "clip", "embedding")):
                continue
            models.append(model_id)
    return models


def _prioritize_models(models: list[str], preferred_order: list[str]) -> list[str]:
    seen = set()
    ordered = []
    # 1. Preferred models in priority order
    for pref in preferred_order:
        for m in models:
            if m.lower() == pref.lower() and m not in seen:
                ordered.append(m)
                seen.add(m)
    # 2. Remaining models
    for m in models:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


def get_available_models_for_provider(provider: str | None = None) -> list[str]:
    """Return the available models for the given provider."""
    normalized = _normalize_provider(provider)
    if not normalized:
        return []

    fast_models = {
        "groq": [GROQ_MODEL or "qwen/qwen3.8-27b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
        "gemini": [GEMINI_MODEL or "gemini-2.0-flash", "gemini-1.5-flash"],
        "mistral": [MISTRAL_MODEL or "mistral-small-latest", "mistral-large-latest"],
    }
    models = fast_models.get(normalized, [])
    # deduplicate while preserving order
    return list(dict.fromkeys(m for m in models if m))


def get_default_provider_and_model(
    provider: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    resolved_provider = _normalize_provider(provider) or LLM_PROVIDER or "groq"
    if resolved_provider not in AVAILABLE_LLM_PROVIDERS:
        raise RuntimeError(
            f"Unknown LLM provider '{resolved_provider}'. Choose one of: {list(AVAILABLE_LLM_PROVIDERS)}"
        )

    if model:
        return resolved_provider, model

    available_models = get_available_models_for_provider(resolved_provider)
    configured_model = _DEFAULT_MODELS.get(resolved_provider, "")

    if configured_model and configured_model in available_models:
        return resolved_provider, configured_model

    if available_models:
        return resolved_provider, available_models[0]

    if configured_model:
        return resolved_provider, configured_model

    raise RuntimeError(f"No model is available for provider '{resolved_provider}'.")


def _format_openai_messages(
    system_prompt: str,
    user_prompt: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Format messages array for OpenAI-compatible chat completion endpoints."""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            content = (turn.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _format_gemini_contents(
    user_prompt: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Format contents array for Google Gemini generateContent endpoint.
    Ensures correct role alternation ('user' <-> 'model') starting with 'user'.
    """
    contents: list[dict[str, Any]] = []
    if history:
        for turn in history:
            role = "model" if turn.get("role") == "assistant" else "user"
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"][0]["text"] += "\n" + content
            else:
                contents.append({"role": role, "parts": [{"text": content}]})

    if contents and contents[0]["role"] != "user":
        contents.pop(0)

    if contents and contents[-1]["role"] == "user":
        contents[-1]["parts"][0]["text"] += f"\n\n{user_prompt}"
    else:
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})

    return contents


def _clean_response_text(text: str) -> str:
    import re
    # Strip thinking blocks from reasoning models if present
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned or text.strip()


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    model_name: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    effective_model = model_name or GROQ_MODEL
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in the environment (.env file).")
    if not effective_model:
        raise RuntimeError("A Groq model is required. Select a model in the UI or set GROQ_MODEL.")

    messages = _format_openai_messages(system_prompt, user_prompt, history)

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": effective_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        timeout=35,
    )
    if response.status_code == 404 and effective_model != "qwen/qwen3.8-27b":
        # Fallback to known available chat model on Groq
        return _call_groq(system_prompt, user_prompt, max_tokens, "qwen/qwen3.8-27b", history)

    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return _clean_response_text(raw)


def _call_mistral(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    model_name: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    effective_model = model_name or MISTRAL_MODEL
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is not set in the environment (.env file).")
    if not effective_model:
        raise RuntimeError("A Mistral model is required. Select a model in the UI or set MISTRAL_MODEL.")

    messages = _format_openai_messages(system_prompt, user_prompt, history)

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": effective_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        timeout=35,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    model_name: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    effective_model = model_name or GEMINI_MODEL
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment (.env file).")
    if not effective_model:
        raise RuntimeError("A Gemini model is required. Select a model in the UI or set GEMINI_MODEL.")

    contents = _format_gemini_contents(user_prompt, history)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{effective_model}:generateContent?key={GEMINI_API_KEY}"
    )
    response = requests.post(
        url,
        json={
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
        },
        timeout=35,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


_PROVIDERS = {
    "groq": _call_groq,
    "mistral": _call_mistral,
    "gemini": _call_gemini,
}


def complete(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 750,
    provider: str | None = None,
    model: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Generic chat completion with multi-turn history support."""
    resolved_provider, resolved_model = get_default_provider_and_model(provider=provider, model=model)
    provider_fn = _PROVIDERS.get(resolved_provider)
    if provider_fn is None:
        raise RuntimeError(
            f"Unknown LLM provider '{resolved_provider}'. Choose one of: {list(_PROVIDERS)}"
        )
    return provider_fn(system_prompt, user_prompt, max_tokens, resolved_model, history)


def _build_user_prompt(question: str, context: str) -> str:
    if not context:
        return (
            f"Customer question: {question}\n\n"
            "Note: No matching phones were found in the catalog for this specific query."
        )
    return f"Matching phones from the catalog:\n\n{context}\n\nCustomer question: {question}"


def _split_answer_and_suggestions(raw_text: str, question: str) -> tuple[str, list[str]]:
    import re
    suggestions: list[str] = []

    if "---SUGGESTIONS---" in raw_text:
        parts = raw_text.split("---SUGGESTIONS---", 1)
        answer = parts[0].strip()
        sug_part = parts[1].strip()
        for line in sug_part.splitlines():
            line = line.strip()
            if line.startswith(("-", "*", "•", "1.", "2.", "3.", ">")):
                cleaned = re.sub(r"^[\-\*•\d\.\s>]+", "", line).strip()
                cleaned = cleaned.strip('"\'[]')
                if cleaned and len(cleaned) > 3:
                    suggestions.append(cleaned)
    else:
        answer = raw_text.strip()

    # Clean up any trailing suggestion headers from the main answer text
    answer = re.sub(r"\n+(?:###?\s*)?(?:Suggested Questions|Follow-up Questions|Next Questions).*?$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()

    # Fallback to smart English suggestions if fewer than 2 were extracted
    if len(suggestions) < 2:
        suggestions = [
            "Which one has the best camera quality?",
            "How is the battery backup and charging speed?",
            "Are there other strong options in this budget?",
        ]

    return answer, suggestions[:4]


def generate_answer(
    question: str,
    context: str,
    provider: str | None = None,
    model: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[str]]:
    """Generate a grounded, natural shopping-assistant reply and predicted follow-up questions."""
    system_prompt = ASSISTANT_SYSTEM_PROMPT.format(context=context) if "{context}" in ASSISTANT_SYSTEM_PROMPT else f"{ASSISTANT_SYSTEM_PROMPT}\n\nPHONE CATALOG DATA:\n{context}"
    raw_reply = complete(
        system_prompt,
        f"Customer query: {question}",
        max_tokens=800,
        provider=provider,
        model=model,
        history=history,
    )
    return _split_answer_and_suggestions(raw_reply, question)
