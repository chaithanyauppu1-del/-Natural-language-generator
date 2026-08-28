
import time
import logging
from typing import List, Dict, Any
from flask import Flask, render_template, request, jsonify

from config import (
    SUPPORTED_STYLES,
    MAX_HISTORY_ITEMS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
    DEFAULT_TEMPERATURE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_NO_REPEAT_NGRAM_SIZE
)

from src.generator import NLGGenerator
from src.evaluation import calculate_evaluation_metrics, get_text_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NLGApp")

app = Flask(__name__)

SESSION_HISTORY: List[Dict[str, Any]] = []

_generator_instance = None


def get_generator():
    global _generator_instance
    if _generator_instance is None:
        logger.info("Initializing T5 Generator model instance...")
        _generator_instance = NLGGenerator.get_instance()
    return _generator_instance


@app.before_request
def preload_model():
    get_generator()


@app.route("/")
def index():
    return render_template(
        "index.html",
        styles=SUPPORTED_STYLES,
        defaults={
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "min_new_tokens": DEFAULT_MIN_NEW_TOKENS,
            "num_beams": DEFAULT_NUM_BEAMS,
            "temperature": DEFAULT_TEMPERATURE,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "no_repeat_ngram_size": DEFAULT_NO_REPEAT_NGRAM_SIZE
        }
    )


@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        data = request.get_json(force=True, silent=True) or request.form.to_dict()
        if not data:
            return jsonify({"success": False, "error": "Invalid payload. Expected JSON body."}), 400

        prompt_text = data.get("prompt_text")
        structured_data = data.get("structured_data")
        style = data.get("style", "general")
        reference_text = data.get("reference_text")

        max_new_tokens = int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        min_new_tokens = int(data.get("min_new_tokens", DEFAULT_MIN_NEW_TOKENS))
        num_beams = int(data.get("num_beams", DEFAULT_NUM_BEAMS))
        temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
        repetition_penalty = float(data.get("repetition_penalty", DEFAULT_REPETITION_PENALTY))
        no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE))

        generator = get_generator()
        result = generator.generate(
            prompt_text=prompt_text,
            structured_data=structured_data,
            style=style,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        generated_text = result["generated_text"]
        statistics = get_text_statistics(generated_text)
        evaluation = calculate_evaluation_metrics(generated_text, reference_text=reference_text)

        history_item = {
            "id": int(time.time() * 1000),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_text": prompt_text,
            "structured_data": structured_data,
            "full_prompt": result["full_prompt"],
            "style": style,
            "generated_text": generated_text,
            "execution_time_sec": result["execution_time_sec"],
            "fact_coverage": result.get("fact_coverage"),
            "statistics": statistics,
            "evaluation": evaluation if evaluation.get("metrics_available") else None
        }

        SESSION_HISTORY.insert(0, history_item)
        if len(SESSION_HISTORY) > MAX_HISTORY_ITEMS:
            SESSION_HISTORY.pop()

        return jsonify({
            "success": True,
            "result": result,
            "fact_coverage": result.get("fact_coverage"),
            "statistics": statistics,
            "evaluation": evaluation,
            "history_id": history_item["id"]
        })

    except ValueError as val_err:
        logger.warning(f"Validation error in /generate: {val_err}")
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as err:
        logger.error(f"Error executing text generation: {err}", exc_info=True)
        return jsonify({"success": False, "error": f"Text generation failed: {str(err)}"}), 500


@app.route("/evaluate", methods=["POST"])
def evaluate_text():
    try:
        data = request.get_json(force=True, silent=True) or request.form.to_dict()
        if not data or "generated_text" not in data:
            return jsonify({"success": False, "error": "Missing required field 'generated_text'."}), 400

        generated_text = data.get("generated_text", "")
        reference_text = data.get("reference_text", "")

        evaluation = calculate_evaluation_metrics(generated_text, reference_text)
        return jsonify({"success": True, "evaluation": evaluation})

    except Exception as err:
        logger.error(f"Error evaluating text: {err}", exc_info=True)
        return jsonify({"success": False, "error": f"Evaluation failed: {str(err)}"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({
        "success": True,
        "history": SESSION_HISTORY,
        "count": len(SESSION_HISTORY)
    })


@app.route("/clear", methods=["POST"])
def clear_history():
    global SESSION_HISTORY
    SESSION_HISTORY = []
    return jsonify({
        "success": True,
        "message": "Session history cleared successfully."
    })


@app.route("/health", methods=["GET"])
def health_check():
    generator = get_generator()
    return jsonify({
        "status": "healthy",
        "model_name": generator.model_name,
        "device": str(generator.device),
        "history_count": len(SESSION_HISTORY)
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error occurred."}), 500


if __name__ == "__main__":
    logger.info("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
