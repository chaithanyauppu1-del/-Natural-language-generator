
import re
from typing import Dict, Any, Optional

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


import logging
logger = logging.getLogger("NLGEvaluation")

METRIC_LIMITATIONS_DISCLAIMER = (
    "Note on Metric Limitations: ROUGE and BLEU metrics measure n-gram lexical overlap between the generated text "
    "and a ground-truth reference text. High scores indicate lexical similarity, but do not guarantee semantic accuracy, "
    "factual correctness, or natural human readability. Automated metrics should be interpreted alongside human qualitative assessment."
)


def get_text_statistics(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0
        }

    char_count = len(text)
    words = text.split()
    word_count = len(words)

    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
            sentence_count = len(sentences)
        except Exception:
            sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
            sentence_count = max(1, len(sentences))
    else:
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = max(1, len(sentences))

    avg_word_length = round(sum(len(w) for w in words) / max(1, word_count), 2)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length
    }


def normalize_text_for_eval(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


from collections import Counter


def _lcs_length(x: list, y: list) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def calculate_pure_python_rouge(ref_clean: str, gen_clean: str) -> Dict[str, Any]:
    ref_tokens = re.findall(r'\b\w+\b', ref_clean.lower())
    gen_tokens = re.findall(r'\b\w+\b', gen_clean.lower())

    if not ref_tokens or not gen_tokens:
        empty_score = {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}
        return {"rouge1": empty_score, "rouge2": empty_score, "rougeL": empty_score}

    def _ngram_score(n: int) -> Dict[str, float]:
        if len(ref_tokens) < n or len(gen_tokens) < n:
            return {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}

        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        gen_ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens) - n + 1)]

        ref_counts = Counter(ref_ngrams)
        gen_counts = Counter(gen_ngrams)

        overlap = sum((ref_counts & gen_counts).values())
        prec = overlap / len(gen_ngrams) if len(gen_ngrams) > 0 else 0.0
        rec = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return {"fmeasure": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)}

    lcs_len = _lcs_length(ref_tokens, gen_tokens)
    prec_l = lcs_len / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
    rec_l = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1_l = (2 * prec_l * rec_l / (prec_l + rec_l)) if (prec_l + rec_l) > 0 else 0.0
    rouge_l = {"fmeasure": round(f1_l, 4), "precision": round(prec_l, 4), "recall": round(rec_l, 4)}

    return {
        "rouge1": _ngram_score(1),
        "rouge2": _ngram_score(2),
        "rougeL": rouge_l
    }


def calculate_evaluation_metrics(
    generated_text: str,
    reference_text: Optional[str] = None
) -> Dict[str, Any]:
    gen_stats = get_text_statistics(generated_text)

    response = {
        "generated_statistics": gen_stats,
        "metrics_available": False,
        "metrics": None,
        "disclaimer": METRIC_LIMITATIONS_DISCLAIMER
    }

    if not reference_text or not reference_text.strip():
        response["message"] = "No reference text provided. ROUGE and BLEU evaluation require a reference text."
        return response

    ref_raw = reference_text.strip()
    gen_raw = generated_text.strip()
    ref_stats = get_text_statistics(ref_raw)
    response["reference_statistics"] = ref_stats

    ref_clean = normalize_text_for_eval(ref_raw)
    gen_clean = normalize_text_for_eval(gen_raw)

    ref_toks = ref_clean.split()
    gen_toks = gen_clean.split()

    logger.info(f"[EVALUATION DEBUG] RAW REFERENCE: {ref_raw!r}")
    logger.info(f"[EVALUATION DEBUG] RAW GENERATED: {gen_raw!r}")
    logger.info(f"[EVALUATION DEBUG] NORMALIZED REFERENCE: {ref_clean!r}")
    logger.info(f"[EVALUATION DEBUG] NORMALIZED GENERATED: {gen_clean!r}")
    logger.info(f"[EVALUATION DEBUG] TOKENIZED REFERENCE: {ref_toks}")
    logger.info(f"[EVALUATION DEBUG] TOKENIZED GENERATED: {gen_toks}")

    scores = {}
    rouge_calculated = False

    if ROUGE_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_results = scorer.score(ref_clean, gen_clean)

            logger.info(f"[EVALUATION DEBUG] RAW ROUGE-1 RESULT: {rouge_results['rouge1']}")
            logger.info(f"[EVALUATION DEBUG] RAW ROUGE-2 RESULT: {rouge_results['rouge2']}")
            logger.info(f"[EVALUATION DEBUG] RAW ROUGE-L RESULT: {rouge_results['rougeL']}")

            scores["rouge"] = {
                "rouge1": {
                    "fmeasure": round(rouge_results["rouge1"].fmeasure, 4),
                    "precision": round(rouge_results["rouge1"].precision, 4),
                    "recall": round(rouge_results["rouge1"].recall, 4)
                },
                "rouge2": {
                    "fmeasure": round(rouge_results["rouge2"].fmeasure, 4),
                    "precision": round(rouge_results["rouge2"].precision, 4),
                    "recall": round(rouge_results["rouge2"].recall, 4)
                },
                "rougeL": {
                    "fmeasure": round(rouge_results["rougeL"].fmeasure, 4),
                    "precision": round(rouge_results["rougeL"].precision, 4),
                    "recall": round(rouge_results["rougeL"].recall, 4)
                }
            }
            rouge_calculated = True
            logger.info(f"[EVALUATION DEBUG] FINAL ROUGE SCORES (rouge-score library): {scores['rouge']}")
        except Exception as e:
            logger.warning(f"[EVALUATION DEBUG] Primary ROUGE scorer exception: {e}. Falling back to pure Python implementation.")

    if not rouge_calculated:
        fallback_rouge = calculate_pure_python_rouge(ref_clean, gen_clean)
        scores["rouge"] = fallback_rouge
        logger.info(f"[EVALUATION DEBUG] RAW ROUGE-1 RESULT (Fallback): {fallback_rouge['rouge1']}")
        logger.info(f"[EVALUATION DEBUG] RAW ROUGE-2 RESULT (Fallback): {fallback_rouge['rouge2']}")
        logger.info(f"[EVALUATION DEBUG] RAW ROUGE-L RESULT (Fallback): {fallback_rouge['rougeL']}")
        logger.info(f"[EVALUATION DEBUG] FINAL ROUGE SCORES (Fallback): {scores['rouge']}")

    if NLTK_AVAILABLE:
        try:
            ref_tokens = [gen_toks if not ref_toks else ref_toks]
            gen_tokens = gen_toks
            smooth_fn = SmoothingFunction().method1

            b1 = sentence_bleu(ref_tokens, gen_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth_fn)
            b2 = sentence_bleu(ref_tokens, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
            b3 = sentence_bleu(ref_tokens, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
            b4 = sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

            scores["bleu"] = {
                "bleu_1": round(b1, 4),
                "bleu_2": round(b2, 4),
                "bleu_3": round(b3, 4),
                "bleu_4": round(b4, 4),
                "cumulative": round(b4, 4)
            }
        except Exception as e:
            scores["bleu_error"] = str(e)

    response["metrics_available"] = True
    response["metrics"] = scores
    return response

