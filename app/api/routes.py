"""Route handlers for prompt optimization endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.models.schemas import OptimizePromptRequest
from app.services.prompt_optimizer import PromptOptimizerService

api_blueprint = Blueprint("api", __name__)
prompt_optimizer_service = PromptOptimizerService()


@api_blueprint.get("/")
def home():
    """Render the Prompt Optimizer web application."""
    return render_template("index.html")


@api_blueprint.get("/health")
def health():
    """Return a simple service health payload."""
    return jsonify({"status": "ok"}), 200


@api_blueprint.get("/optimize_prompt")
def optimize_prompt_help():
    """Explain correct usage when the API endpoint is opened in a browser."""
    return (
        jsonify(
            {
                "message": "Use POST /optimize_prompt with a JSON body.",
                "example": {
                    "prompt": "Explain how to optimize a Python function for performance",
                    "mode": "cost",
                },
            }
        ),
        200,
    )


@api_blueprint.post("/optimize_prompt")
def optimize_prompt():
    """Accept a prompt optimization request and return the optimized result."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    request_data = OptimizePromptRequest.from_dict(payload)
    if not request_data.prompt.strip():
        return jsonify({"error": "The 'prompt' field is required."}), 400

    if request_data.mode not in {"cost", "context"}:
        return jsonify({"error": "The 'mode' field must be either 'cost' or 'context'."}), 400

    result = prompt_optimizer_service.optimize_prompt(payload=payload)
    status_code = 200 if result.get("pipeline_status") == "success" else 500
    return jsonify(result), status_code
