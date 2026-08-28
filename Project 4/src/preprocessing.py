
import re
import json
from typing import Dict, Any, Optional, Union
from config import SUPPORTED_STYLES, MAX_PROMPT_LENGTH


def clean_text(text: Optional[str]) -> str:
    if text is None:
        raise ValueError("Input text cannot be None.")

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = ' '.join(lines)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if not cleaned:
        raise ValueError("Input text cannot be empty or contain only whitespace.")

    return cleaned


def parse_structured_input(raw_data: Union[str, Dict[str, Any], list]) -> str:
    if isinstance(raw_data, str):
        raw_data_clean = raw_data.strip()
        if raw_data_clean.startswith('{') or raw_data_clean.startswith('['):
            try:
                parsed = json.loads(raw_data_clean)
                return parse_structured_input(parsed)
            except json.JSONDecodeError:
                pass

        lines = [line.strip() for line in raw_data_clean.splitlines() if line.strip()]
        kv_pairs = []
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                kv_pairs.append(f"{key.strip()}: {val.strip()}")
            elif '=' in line:
                key, val = line.split('=', 1)
                kv_pairs.append(f"{key.strip()}: {val.strip()}")
            else:
                kv_pairs.append(line)
        return "; ".join(kv_pairs)

    elif isinstance(raw_data, dict):
        kv_pairs = [f"{str(k).strip()}: {str(v).strip()}" for k, v in raw_data.items()]
        return "; ".join(kv_pairs)

    elif isinstance(raw_data, list):
        items = [parse_structured_input(item) if isinstance(item, (dict, list)) else str(item).strip() for item in raw_data]
        return ", ".join(items)

    return str(raw_data).strip()


def extract_structured_facts(raw_data: Optional[Union[str, Dict[str, Any], list]]) -> list:
    if not raw_data:
        return []

    facts = []

    def make_fact(k: str, v: str) -> dict:
        k_clean = str(k).strip()
        v_clean = str(v).strip()
        return {
            "key": k_clean,
            "value": v_clean,
            "label": k_clean.lower(),
            "raw_str": f"{k_clean}: {v_clean}" if k_clean.lower() != v_clean.lower() else v_clean
        }

    if isinstance(raw_data, str):
        raw_clean = raw_data.strip()
        if raw_clean.startswith('{') or raw_clean.startswith('['):
            try:
                parsed = json.loads(raw_clean)
                return extract_structured_facts(parsed)
            except Exception:
                pass

        lines = [line.strip() for line in raw_clean.splitlines() if line.strip()]
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                if key.strip() and val.strip():
                    facts.append(make_fact(key, val))
            elif '=' in line:
                key, val = line.split('=', 1)
                if key.strip() and val.strip():
                    facts.append(make_fact(key, val))
            else:
                tokens = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?%?|\b[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})*\b', line)
                for tok in tokens:
                    if tok.strip():
                        facts.append(make_fact(tok, tok))

    elif isinstance(raw_data, dict):
        for k, v in raw_data.items():
            if str(k).strip() and str(v).strip():
                facts.append(make_fact(k, v))

    elif isinstance(raw_data, list):
        for item in raw_data:
            facts.extend(extract_structured_facts(item))

    seen = set()
    dedup = []
    for f in facts:
        fact_id = (f["key"].lower(), f["value"].lower())
        if fact_id not in seen:
            seen.add(fact_id)
            dedup.append(f)

    return dedup


def build_generation_prompt(
    prompt_text: Optional[str] = None,
    structured_data: Optional[Union[str, Dict[str, Any]]] = None,
    style: str = "general"
) -> str:
    has_prompt = prompt_text is not None and bool(prompt_text.strip())
    has_structured = structured_data is not None and bool(str(structured_data).strip())

    if not has_prompt and not has_structured:
        raise ValueError("Please provide either a prompt instruction or structured data.")

    style_key = style.lower() if style and style.lower() in SUPPORTED_STYLES else "general"
    prefix = SUPPORTED_STYLES[style_key]["prefix"]

    lines = [prefix]
    if has_structured:
        formatted_struct = parse_structured_input(structured_data)
        lines.append(f"Facts: {formatted_struct}")
        lines.append("Instructions: Preserve exact key labels and numeric values semantically without substituting terms or inferring unstated events.")

    if has_prompt:
        clean_p = clean_text(prompt_text)
        lines.append(f"Task: {clean_p}")

    full_prompt = "\n".join(lines)

    if len(full_prompt) > MAX_PROMPT_LENGTH:
        full_prompt = full_prompt[:MAX_PROMPT_LENGTH]

    return full_prompt


