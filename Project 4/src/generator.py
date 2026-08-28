
import re
import time
import logging
import threading
from typing import Dict, Any, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
    DEFAULT_TEMPERATURE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_NO_REPEAT_NGRAM_SIZE
)
from src.preprocessing import build_generation_prompt, extract_structured_facts, parse_structured_input

import re
import time
import logging
from typing import Dict, Any, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
    DEFAULT_TEMPERATURE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_NO_REPEAT_NGRAM_SIZE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NLGGenerator")


def extract_core_key_tokens_and_stems(key_str: str) -> dict:
    key_lower = key_str.lower().strip()
    tokens = re.findall(r'\b\w+\b', key_lower)

    stop_words = {'is', 'of', 'the', 'a', 'an', 'for', 'in', 'by', 'with', 'and', 'or', 'to', 'on', 'at', 'about'}
    generic_cat_words = {
        'rate', 'percentage', 'percent', 'ratio', 'index', 'count', 'number', 'value', 'level',
        'metric', 'amount', 'total', 'time', 'duration', 'period', 'year', 'date', 'area',
        'efficiency', 'accuracy', 'reduction', 'satisfaction', 'capacity', 'volume', 'throughput',
        'margin', 'growth', 'budget', 'revenue', 'funding', 'cost', 'price', 'savings', 'score'
    }

    filtered_tokens = [t for t in tokens if t not in stop_words]
    core_tokens = [t for t in filtered_tokens if t not in generic_cat_words]
    if not core_tokens:
        core_tokens = filtered_tokens

    stems = set()
    for t in core_tokens:
        stems.add(t)
        if len(t) > 3:
            if t.endswith('s'):
                stems.add(t[:-1])
            if t.endswith('age'):
                stems.add(t)
                stems.add(t[:-3] + 's')
                stems.add(t[:-3] + 'ed')
                stems.add(t[:-3] + 'ing')
            elif 'ation' in t or 'tion' in t:
                stems.add(t[:-5])
                stems.add(t[:-4])
                if 'filtration' in t or 'filter' in t:
                    stems.add('filtration')
                    stems.add('filter')
                    stems.add('filters')
                    stems.add('filtering')
                stems.add(t[:-3] + 'es')
                stems.add(t[:-3] + 'ed')
            elif t.endswith('ment'):
                stems.add(t[:-4])
                stems.add(t[:-4] + 's')
                stems.add(t[:-4] + 'ed')
                stems.add(t[:-4] + 'ing')
            elif t.endswith('ation'):
                stems.add(t[:-5] + 'er')
                stems.add(t[:-5] + 'ers')
                stems.add(t[:-5] + 'ering')
                stems.add(t[:-5] + 'ered')
                stems.add(t[:-5] + 'e')
                stems.add(t[:-5] + 'es')
                stems.add(t[:-5] + 'ed')
                stems.add(t[:-5] + 'ing')
                stems.add(t[:-4])
                stems.add(t[:-5])
            elif t.endswith('tion') or t.endswith('sion'):
                stems.add(t[:-4] + 'e')
                stems.add(t[:-4] + 'ed')
                stems.add(t[:-4] + 'es')
                stems.add(t[:-4] + 'ing')
                stems.add(t[:-3])
            elif t.endswith('ity'):
                stems.add(t[:-3] + 'e')
                stems.add(t[:-3])
            elif 'employ' in t:
                stems.add('employ')
                stems.add('employs')
                stems.add('employed')
                stems.add('employing')
                stems.add('employee')
                stems.add('employees')
            elif 'headquarter' in t or 'location' in t or 'hq' in t:
                stems.add('headquarters')
                stems.add('headquarter')
                stems.add('headquartered')
                stems.add('based')
                stems.add('located')
                stems.add('location')
                stems.add('in')
                stems.add('at')
            elif t.endswith('ed'):
                stems.add(t[:-2])
                stems.add(t[:-1])

    return {
        "key_lower": key_lower,
        "tokens": tokens,
        "core_tokens": core_tokens,
        "stems": stems,
        "generic_cat_words": set(tokens) & generic_cat_words
    }


def is_semantic_key_matched(key_str: str, val_str: str, sentence: str, gen_lower: str) -> bool:
    s_lower = sentence.lower()
    key_lower = key_str.lower().strip()
    val_lower = val_str.lower().strip()

    if re.search(rf'{re.escape(key_str)}\s*:\s*{re.escape(val_str)}', sentence, re.IGNORECASE):
        return not sentence.strip().startswith(f"{key_lower}:")

    entity_keys = {
        "entity", "project", "company", "organization", "firm", "business", "department", "dept",
        "university", "college", "institution", "product", "system", "tool", "software",
        "service", "initiative", "hospital", "app", "application", "dataset", "model",
        "study", "group", "team", "client", "name", "title"
    }

    if key_lower in entity_keys:
        if val_lower in s_lower or val_lower in gen_lower:
            return True

    analysis = extract_core_key_tokens_and_stems(key_str)
    core_tokens = analysis["core_tokens"]
    stems = analysis["stems"]

    if "year" in key_lower or "date" in key_lower:
        unsupported_events = {
            "launched", "launch", "bought", "sold", "acquired", "shipped", "started", "introduced",
            "scheduled", "expected", "planned", "slated", "intended"
        }
        for ev in unsupported_events:
            if ev in s_lower and ev not in key_lower:
                return False
        if re.search(r'\b(?:scheduled|expected|planned|slated|intended|set)\s+to\b', s_lower):
            return False

    for core in core_tokens:
        core_stem_present = any(st in s_lower for st in stems if len(st) >= 3)
        if not core_stem_present:
            if core == "energy" and "power" in s_lower:
                return False
            if core == "coverage" and "total" in s_lower and "cover" not in s_lower:
                return False
            if core == "pass" and ("placement" in s_lower or "job" in s_lower or "recruit" in s_lower):
                return False
            if core == "placement" and ("pass" in s_lower or "exam" in s_lower or "academic" in s_lower):
                return False

    has_stem_match = any(st in s_lower for st in stems)

    if not has_stem_match:
        if "student" in key_lower and any(w in s_lower for w in ["student", "pupil", "enrol"]):
            has_stem_match = True
        elif "faculty" in key_lower and any(w in s_lower for w in ["faculty", "teacher", "staff", "professor"]):
            has_stem_match = True
        elif "employee" in key_lower and any(w in s_lower for w in ["employee", "staff", "worker"]):
            has_stem_match = True
        elif "headquarter" in key_lower and any(w in s_lower for w in ["headquarter", "headquartered", "location", "based"]):
            has_stem_match = True
        elif "establish" in key_lower or "found" in key_lower:
            if any(w in s_lower for w in ["establish", "founded", "since", "established"]):
                has_stem_match = True
        elif "subject" in key_lower or "course" in key_lower:
            if any(w in s_lower for w in ["subject", "course", "top", "main"]):
                has_stem_match = True
        elif "dept" in key_lower or "department" in key_lower:
            if any(w in s_lower for w in ["dept", "department", "ai"]):
                has_stem_match = True

    return has_stem_match


def check_fact_coverage(facts: list, generated_text: str) -> dict:
    if not facts or not generated_text:
        return {
            "complete": True,
            "covered": True,
            "coverage": "0/0",
            "coverage_percentage": 100.0,
            "semantic_coverage": True,
            "missing_facts": [],
            "missing_key_values": [],
            "covered_facts": [],
            "total_facts": 0,
            "covered_count": 0,
            "fact_details": []
        }

    norm_facts = []
    for f in facts:
        if isinstance(f, dict):
            norm_facts.append(f)
        else:
            f_str = str(f).strip()
            norm_facts.append({
                "key": f_str,
                "value": f_str,
                "label": f_str.lower(),
                "raw_str": f_str
            })

    gen_lower = generated_text.lower()
    sentences = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)|[!?\n;]+', gen_lower) if s.strip()]

    fact_details = []
    covered_facts = []
    missing_facts = []
    missing_key_values = []

    for fact in norm_facts:
        val_str = str(fact.get("value", "")).strip()
        key_str = str(fact.get("key", "")).strip()
        raw_str = str(fact.get("raw_str", f"{key_str}: {val_str}")).strip()

        if not val_str:
            continue

        val_lower = val_str.lower()
        key_lower = key_str.lower()
        val_uncomma = val_lower.replace(',', '')

        value_present = False
        matching_sentences = []
        if val_lower.endswith('%'):
            num_part = val_lower[:-1].strip()
            num_uncomma = num_part.replace(',', '')
            pct_pattern = rf'\b(?:{re.escape(num_part)}|{re.escape(num_uncomma)})\s*(?:%|(?:percent|per\s*cent)\b)'
            value_present = bool(re.search(pct_pattern, gen_lower))
            if value_present:
                matching_sentences = [s for s in sentences if num_part in s or num_uncomma in s or re.search(pct_pattern, s)]
        elif re.fullmatch(r'\d+(?:,\d+)*(?:\.\d+)?', val_lower):
            pattern = rf'\b(?:{re.escape(val_lower)}|{re.escape(val_uncomma)})(?!\s*%)\b'
            value_present = bool(re.search(pattern, gen_lower))
            if value_present:
                matching_sentences = [s for s in sentences if val_lower in s or val_uncomma in s or re.search(pattern, s)]
        else:
            val_norm = val_lower.replace('²', '2').replace('³', '3')
            val_uncomma_norm = val_uncomma.replace('²', '2').replace('³', '3')
            gen_norm = gen_lower.replace('²', '2').replace('³', '3')
            value_present = val_lower in gen_lower or val_uncomma in gen_lower or val_norm in gen_norm or val_uncomma_norm in gen_norm
            if value_present:
                matching_sentences = [s for s in sentences if val_lower in s or val_uncomma in s or val_norm in s.replace('²', '2').replace('³', '3')]
            else:
                words = [w for w in re.findall(r'\b\w+\b', val_norm) if len(w) > 2]
                if words and all(w in gen_norm for w in words):
                    value_present = True
                    matching_sentences = [s for s in sentences if any(w in s.replace('²', '2').replace('³', '3') for w in words)]

        semantic_match = False
        if value_present:
            if not matching_sentences:
                matching_sentences = sentences

            for s in matching_sentences:
                if is_semantic_key_matched(key_str, val_str, s, gen_lower):
                    semantic_match = True
                    break

        fact_detail = {
            "key": key_str,
            "value": val_str,
            "raw_str": raw_str,
            "value_present": value_present,
            "semantic_match": semantic_match
        }
        fact_details.append(fact_detail)

        if value_present and semantic_match:
            covered_facts.append(val_str)
        else:
            missing_facts.append(val_str)
            missing_key_values.append(raw_str)

    total_count = len(norm_facts)
    covered_count = total_count - len(missing_facts)
    is_complete = len(missing_facts) == 0
    semantic_coverage = is_complete

    if re.search(r'\b(?:shows|reports|indicates)\s+(?:uses|has|was|is|recorded|employs)\b', gen_lower):
        is_complete = False
        semantic_coverage = False

    if re.search(r'\bhas\s+(?:implementation|completion|turnaround|processing|annual)\s+(?:time|year|date|savings|reduction|rate|budget|capacity|duration)\s+of\b', gen_lower):
        is_complete = False
        semantic_coverage = False

    if re.search(r'\b(?:\d{4}\s+completion\s+year|has\s+\d{4}\s+completion)\b', gen_lower):
        is_complete = False
        semantic_coverage = False

    if re.search(r'\bcover\s+area\b', gen_lower):
        is_complete = False
        semantic_coverage = False

    if re.search(r'\b(?:scheduled|expected|planned|slated|intended)\s+to\s+be\s+(?:released|launched|introduced)\b', gen_lower):
        is_complete = False
        semantic_coverage = False

    if norm_facts:
        explicit_entity = None
        for f in norm_facts:
            k = f.get("key", "").lower()
            v = f.get("value", "")
            if k in {"project", "company", "organization", "university", "system", "product", "initiative", "hospital", "app", "model", "study", "entity"} and v:
                explicit_entity = v.lower()
                break
        if explicit_entity and "the report" in gen_lower:
            if re.search(r'\bthe\s+report\s+(?:has|uses|includes|shows|reports)\b', gen_lower):
                is_complete = False
                semantic_coverage = False

    coverage_pct = round((covered_count / total_count * 100.0), 1) if total_count > 0 else 100.0

    return {
        "complete": is_complete,
        "covered": is_complete,
        "coverage": f"{covered_count}/{total_count}",
        "coverage_percentage": coverage_pct,
        "semantic_coverage": semantic_coverage,
        "missing_facts": missing_facts,
        "missing_key_values": missing_key_values,
        "covered_facts": covered_facts,
        "total_facts": total_count,
        "covered_count": covered_count,
        "fact_details": fact_details
    }


def detect_entity_subject(structured_data: Optional[str] = None, prompt_text: Optional[str] = None) -> str:
    if structured_data:
        facts = extract_structured_facts(structured_data)
        if facts:
            entity_keys = {
                "project", "company", "organization", "firm", "business", "department", "dept",
                "university", "college", "institution", "product", "system", "tool", "software",
                "service", "initiative", "hospital", "app", "application", "dataset", "model",
                "study", "group", "team", "client", "name", "entity", "title"
            }
            for f in facts:
                if isinstance(f, dict):
                    k = f.get("key", "").strip()
                    v = f.get("value", "").strip()
                    k_lower = k.lower()
                    if k_lower in entity_keys and v:
                        if k_lower in {"department", "dept"}:
                            return f"the {v} department" if "department" not in v.lower() else v
                        return v

            first_fact = facts[0]
            if isinstance(first_fact, dict):
                fk = first_fact.get("key", "").strip()
                fv = first_fact.get("value", "").strip()
                if fv and not fv.replace(',', '').replace('.', '').isdigit() and not fv.startswith('$') and not fv.endswith('%'):
                    if fk.lower() not in {"accuracy", "latency", "version", "status", "date", "year", "revenue", "employees", "students"}:
                        return fv

    if prompt_text:
        match = re.search(r'\b(?:for|about|on)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s\-\_]+?)(?:\s+department|\s+project|\s+system|\s+using|\.|\,|$)', prompt_text)
        if match:
            extracted = match.group(1).strip()
            if extracted and extracted.lower() not in {"the", "a", "an", "provided", "following"}:
                return extracted

    combined = f"{structured_data or ''} {prompt_text or ''}".lower()
    if "project" in combined or "initiative" in combined:
        return "the project"
    elif "department" in combined or "dept" in combined:
        return "the department"
    elif "company" in combined or "firm" in combined or "organization" in combined:
        return "the company"
    elif "university" in combined or "college" in combined:
        return "the university"
    elif "product" in combined or "system" in combined or "software" in combined:
        return "the system"

    return "the report"


from enum import Enum


class SemanticCategory(Enum):
    COUNT = "count"
    PERCENTAGE = "percentage"
    RATE = "rate"
    CURRENCY = "currency"
    DURATION = "duration"
    YEAR = "year"
    LOCATION = "location"
    VERSION = "version"
    CAPACITY = "capacity"
    PERFORMANCE_METRIC = "performance_metric"
    OTHER = "other"


def classify_semantic_category(key: str, val: str) -> SemanticCategory:
    key_lower = key.lower().strip()
    val_lower = val.lower().strip()

    if val.startswith('$') or val.startswith('€') or val.startswith('£') or val.startswith('₹') or any(c in key_lower for c in ["revenue", "sales", "profit", "budget", "funding", "cost", "price", "savings"]):
        return SemanticCategory.CURRENCY

    if val.endswith('%') or any(k in key_lower for k in ["percentage", "percent", "rate", "ratio", "accuracy", "efficiency", "reduction", "growth", "satisfaction", "index", "margin"]):
        return SemanticCategory.PERCENTAGE if val.endswith('%') or "percentage" in key_lower or "reduction" in key_lower or "accuracy" in key_lower or "efficiency" in key_lower or "index" in key_lower else SemanticCategory.RATE

    if "version" in key_lower:
        return SemanticCategory.VERSION

    if any(k in key_lower for k in ["capacity", "volume", "bandwidth", "memory", "storage"]):
        return SemanticCategory.CAPACITY

    if any(k in key_lower for k in ["headquarter", "location", "address"]) or re.search(r'\b(?:city|country)\b', key_lower):
        return SemanticCategory.LOCATION

    if any(k in key_lower for k in ["year", "date", "completion", "established", "founded", "release"]) and (val.isdigit() and len(val) == 4 or "year" in key_lower):
        return SemanticCategory.YEAR

    if any(k in key_lower for k in ["time", "duration", "period", "latency", "delay", "seconds", "minutes", "hours", "days", "weeks", "months", "years"]) and not any(k in key_lower for k in ["completion year", "release year", "target year", "established", "founded"]):
        return SemanticCategory.DURATION

    if any(k in key_lower for k in ["capacity", "volume", "bandwidth", "memory", "storage"]):
        return SemanticCategory.CAPACITY

    if any(k in key_lower for k in ["throughput", "speed", "score", "benchmark"]):
        return SemanticCategory.PERFORMANCE_METRIC

    val_cleaned = val.replace(',', '').replace('.', '')
    if val_cleaned.isdigit() or key_lower.endswith('s') or any(k in key_lower for k in ["count", "number", "students", "employees", "subscribers", "departments", "regions", "units", "items", "members", "languages"]):
        return SemanticCategory.COUNT

    return SemanticCategory.OTHER


def format_semantic_safeguard_clause(fact_dict: dict) -> str:
    if not isinstance(fact_dict, dict):
        return str(fact_dict).strip()

    key = fact_dict.get("key", "").strip()
    val = fact_dict.get("value", "").strip()
    if not key or not val:
        return str(fact_dict).strip()

    key_lower = key.lower()
    article = "an" if key_lower[0] in 'aeiou' else "a"
    category = classify_semantic_category(key, val)

    if category == SemanticCategory.VERSION:
        return f"uses version {val}"

    elif category == SemanticCategory.LOCATION:
        if "headquarter" in key_lower:
            return f"is headquartered in {val}"
        return f"is located in {val}"

    elif category == SemanticCategory.YEAR:
        if "establish" in key_lower or "found" in key_lower:
            return f"was established in {val}" if "establish" in key_lower else f"was founded in {val}"
        return f"has a {key_lower} of {val}"

    elif category in {SemanticCategory.PERCENTAGE, SemanticCategory.RATE, SemanticCategory.DURATION, SemanticCategory.CAPACITY, SemanticCategory.PERFORMANCE_METRIC}:
        if "attend" in key_lower:
            return f"maintains an average attendance of {val}"
        elif "subject" in key_lower or "course" in key_lower:
            return f"features {val} as the top subject"
        return f"has {article} {key_lower} of {val}"

    elif category == SemanticCategory.CURRENCY:
        if "revenue" in key_lower and val.startswith('$'):
            return f"has revenue of {val}"
        return f"has {article} {key_lower} of {val}"

    elif category == SemanticCategory.COUNT:
        if "student" in key_lower:
            return f"has {val} students"
        elif "employee" in key_lower or "staff" in key_lower:
            return f"employs {val} staff members" if "employee" in key_lower else f"has {val} employees"
        elif "faculty" in key_lower or "teacher" in key_lower:
            return f"has {val} faculty members"
        elif "department" in key_lower:
            return f"has {val} departments" if val.replace(',', '').isdigit() else f"includes the {val} department"
        elif "region" in key_lower or "area" in key_lower or "district" in key_lower:
            return f"covers {val} {key_lower}" if val.replace(',', '').isdigit() else f"has {val} {key_lower}"
        elif val.replace(',', '').replace('.', '').isdigit() and key_lower.endswith('s'):
            return f"has {val} {key_lower}"
        else:
            return f"has {article} {key_lower} of {val}"

    else:
        return f"has {article} {key_lower} of {val}"


class NLGGenerator:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing NLGGenerator with model '{self.model_name}' on device '{self.device}'.")

        logger.info(f"Loading primary model weights for '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        logger.info(f"Successfully loaded model '{self.model_name}' on device {self.device}.")

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME) -> 'NLGGenerator':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name)
        return cls._instance

    def generate(
        self,
        prompt_text: Optional[str] = None,
        structured_data: Optional[str] = None,
        style: str = "general",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        num_beams: int = DEFAULT_NUM_BEAMS,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE
    ) -> dict:
        start_time = time.time()
        full_prompt = build_generation_prompt(prompt_text, structured_data, style)

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "early_stopping": True
        }

        if temperature > 0 and temperature != 1.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_kwargs
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        fact_coverage_meta = None
        if structured_data:
            facts = extract_structured_facts(structured_data)
            coverage_res = check_fact_coverage(facts, generated_text)

            logger.info(f"[DEBUG] Attempt 1 Output: '{generated_text}'")
            logger.info(f"[DEBUG] Attempt 1 Semantic Coverage: {coverage_res['coverage']} (semantic_coverage: {coverage_res['semantic_coverage']})")
            logger.info(f"[DEBUG] Missing Semantic Facts: {coverage_res['missing_key_values']}")

            if not coverage_res["semantic_coverage"] and coverage_res["missing_key_values"]:
                missing_set = set(coverage_res["missing_key_values"])
                missing_fact_dicts = [f for f in facts if isinstance(f, dict) and (f.get("raw_str") in missing_set or f.get("value") in coverage_res["missing_facts"])]
                missing_clauses = [format_semantic_safeguard_clause(fd) for fd in missing_fact_dicts]
                missing_clause_str = ", ".join(missing_clauses)
                logger.info(f"Attempt 1 fact coverage incomplete ({coverage_res['coverage']}). Missing: {coverage_res['missing_key_values']}. Running Attempt 2 corrective retry...")

                subject = detect_entity_subject(structured_data, prompt_text)
                retry_prompt = (
                    f"Write a formal report paragraph for {subject} incorporating these details naturally in complete sentences:\n"
                    f"{missing_clause_str}\n\n"
                    f"Requirements:\n"
                    f"1. Write clear, fluent, natural sentences.\n"
                    f"2. Do NOT output raw key-value headers or raw lists.\n"
                    f"3. Every percentage and number must retain its correct label and meaning.\n"
                    f"4. Include all details naturally: {parse_structured_input(structured_data)}"
                )

                retry_inputs = self.tokenizer(
                    retry_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.inference_mode():
                    retry_outputs = self.model.generate(
                        retry_inputs.input_ids,
                        attention_mask=retry_inputs.attention_mask,
                        **gen_kwargs
                    )
                retry_text = self.tokenizer.decode(retry_outputs[0], skip_special_tokens=True).strip()

                retry_coverage = check_fact_coverage(facts, retry_text)
                logger.info(f"[DEBUG] Attempt 2 Output: '{retry_text}'")
                logger.info(f"[DEBUG] Attempt 2 Semantic Coverage: {retry_coverage['coverage']} (semantic_coverage: {retry_coverage['semantic_coverage']})")

                if not coverage_res["complete"] and (retry_coverage["complete"] or retry_coverage["covered_count"] >= coverage_res["covered_count"]):
                    generated_text = retry_text
                    coverage_res = retry_coverage

            if not coverage_res["semantic_coverage"] and coverage_res["missing_key_values"]:
                logger.info(f"Applying label-preserving semantic safeguard for remaining missing facts: {coverage_res['missing_key_values']}")

                missing_set = set(coverage_res["missing_key_values"])
                missing_fact_dicts = [f for f in facts if isinstance(f, dict) and (f.get("raw_str") in missing_set or f.get("value") in coverage_res["missing_facts"])]

                gen_lower = generated_text.lower()
                genuinely_missing_fact_dicts = []
                for fd in missing_fact_dicts:
                    v = str(fd.get("value", "")).strip().lower()
                    if not v:
                        continue
                    v_uncomma = v.replace(',', '')
                    is_present = False
                    if v.endswith('%'):
                        num_part = v[:-1].strip()
                        num_uncomma = num_part.replace(',', '')
                        is_present = num_part in gen_lower or num_uncomma in gen_lower
                    elif re.fullmatch(r'\d+(?:,\d+)*(?:\.\d+)?', v):
                        pattern = rf'\b(?:{re.escape(v)}|{re.escape(v_uncomma)})(?!\s*%)\b'
                        is_present = bool(re.search(pattern, gen_lower))
                    else:
                        v_norm = v.replace('²', '2').replace('³', '3')
                        v_uncomma_norm = v_uncomma.replace('²', '2').replace('³', '3')
                        gen_norm = gen_lower.replace('²', '2').replace('³', '3')
                        is_present = v in gen_lower or v_uncomma in gen_lower or v_norm in gen_norm or v_uncomma_norm in gen_norm
                        if not is_present:
                            words = [w for w in re.findall(r'\b\w+\b', v_norm) if len(w) > 2]
                            if words and all(w in gen_norm for w in words):
                                is_present = True

                    if not is_present:
                        genuinely_missing_fact_dicts.append(fd)

                clauses = [format_semantic_safeguard_clause(fd) for fd in genuinely_missing_fact_dicts]
                logger.info(f"[DEBUG] Final Safeguard Clauses: {clauses}")

                if clauses:
                    subject = detect_entity_subject(structured_data, prompt_text)
                    intro = f"Additionally, {subject}"

                    if len(clauses) == 1:
                        c_str = clauses[0]
                        safeguard_sentence = f" {intro} {c_str}."
                    elif len(clauses) == 2:
                        c1, c2 = clauses[0], clauses[1]
                        if c1.startswith("has ") and c2.startswith("has "):
                            c2_trim = c2[4:]
                            safeguard_sentence = f" {intro} {c1} and {c2_trim}."
                        else:
                            safeguard_sentence = f" {intro} {c1} and {c2}."
                    else:
                        first_c = clauses[0]
                        rest_cs = clauses[1:]
                        if first_c.startswith("has "):
                            trimmed_rest = []
                            for rc in rest_cs:
                                if rc.startswith("has "):
                                    trimmed_rest.append(rc[4:])
                                else:
                                    trimmed_rest.append(rc)
                            clause_body = ", ".join([first_c] + trimmed_rest[:-1]) + f", and {trimmed_rest[-1]}"
                            safeguard_sentence = f" {intro} {clause_body}."
                        else:
                            clause_body = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
                            safeguard_sentence = f" {intro} {clause_body}."

                    generated_text = generated_text.rstrip('.') + '.' + safeguard_sentence
                    coverage_res = check_fact_coverage(facts, generated_text)

            logger.info(f"[DEBUG] Final Returned Output: '{generated_text}'")
            fact_coverage_meta = coverage_res

        elapsed_sec = round(time.time() - start_time, 3)

        return {
            "generated_text": generated_text,
            "full_prompt": full_prompt,
            "prompt_text": prompt_text,
            "structured_data": structured_data,
            "style": style,
            "model_name": self.model_name,
            "device": str(self.device),
            "execution_time_sec": elapsed_sec,
            "fact_coverage": fact_coverage_meta,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "num_beams": num_beams,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size
            }
        }
