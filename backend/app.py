"""Flask API for one-document, web-source plagiarism evidence analysis."""

from __future__ import annotations

import logging
import mimetypes
import uuid
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import text

from backend.config import Settings
from backend.database.database import SessionLocal, initialize_database
from backend.services.database_service import save_web_analysis_to_database
from backend.services.exceptions import ConfigurationError, ExternalServiceError, ModelUnavailableError
from backend.services.history_service import (
    clear_all_analyses,
    delete_analysis_by_id,
    get_all_analyses,
    get_analysis_by_id,
)
from backend.services.plagiarism_pipeline import analyze_web_document
from backend.services.sbert_model import sbert_available


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / "data" / "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_MIME_TYPES = {
    ".pdf": {
        "application/pdf", "application/x-pdf", "application/acrobat",
        "applications/vnd.pdf", "text/pdf", "text/x-pdf", "application/octet-stream",
    },
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/docx", "application/msword", "application/octet-stream", "application/zip",
    },
    ".txt": {
        "text/plain", "text/x-python", "text/markdown", "text/csv",
        "application/octet-stream", "", None,
    },
}


def create_app():
    app = Flask(__name__)
    CORS(app)
    settings = Settings.from_environment()
    app.config["MAX_CONTENT_LENGTH"] = settings.max_file_size_bytes
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    initialize_database()

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"status": "error", "message": str(getattr(error, "description", "Bad request."))}), 400

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"status": "error", "message": "Resource not found."}), 404

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"status": "error", "message": "File exceeds the configured upload size limit."}), 413

    @app.errorhandler(500)
    def server_error(_error):
        return jsonify({"status": "error", "message": "Internal server error occurred."}), 500

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "status": "success", "message": "Explainable Semantic Plagiarism Detection API",
            "workflow": "one document -> public web sources -> evidence report", "version": "2.0",
        })

    @app.route("/api/health", methods=["GET"])
    def health_check():
        database = "ok"
        session = SessionLocal()
        try:
            session.execute(text("SELECT 1"))
        except Exception:
            database = "unavailable"
        finally:
            session.close()
        current_settings = Settings.from_environment()
        return jsonify({
            "status": "healthy" if database == "ok" else "degraded", "database": database,
            "tavily_configured": bool(current_settings.tavily_api_key),
            "firecrawl_configured": bool(current_settings.firecrawl_api_key),
            "sbert_available": sbert_available(),
        })

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        if "document" not in request.files:
            return jsonify({"status": "error", "message": "A document upload is required."}), 400
        uploaded = request.files["document"]
        if not uploaded.filename:
            return jsonify({"status": "error", "message": "The uploaded file has no filename."}), 400
        original_name = Path(uploaded.filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"status": "error", "message": "Only PDF, DOCX, and TXT files are supported."}), 400
        mimetype = uploaded.mimetype or mimetypes.guess_type(original_name)[0]
        if mimetype and mimetype not in ALLOWED_MIME_TYPES[suffix]:
            return jsonify({"status": "error", "message": "The file MIME type does not match its extension."}), 400
        uploaded.stream.seek(0, 2)
        size = uploaded.stream.tell()
        uploaded.stream.seek(0)
        current_settings = Settings.from_environment()
        if size <= 0:
            return jsonify({"status": "error", "message": "The uploaded file is empty."}), 400
        if size > current_settings.max_file_size_bytes:
            return jsonify({"status": "error", "message": "File exceeds the configured upload size limit."}), 413

        temporary_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}{suffix}"
        try:
            uploaded.save(temporary_path)
            LOGGER.info("Starting analysis for a %s upload", suffix)
            report = analyze_web_document(temporary_path, original_name, current_settings)
            report["analysis_id"] = save_web_analysis_to_database(report)
            return jsonify(report), 200
        except ConfigurationError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 422
        except ExternalServiceError as exc:
            LOGGER.warning("%s failed: %s", exc.provider, exc)
            return jsonify({"status": "error", "message": f"{exc.provider} source discovery failed."}), 502
        except ModelUnavailableError:
            LOGGER.exception("SBERT model unavailable")
            return jsonify({"status": "error", "message": "Semantic similarity model is unavailable."}), 503
        except Exception:
            LOGGER.exception("Unexpected analysis failure")
            return jsonify({"status": "error", "message": "An unexpected analysis error occurred."}), 500
        finally:
            temporary_path.unlink(missing_ok=True)

    @app.route("/api/history", methods=["GET"])
    def history():
        try:
            analyses = get_all_analyses()
            return jsonify({"status": "success", "total": len(analyses), "analyses": analyses})
        except Exception:
            LOGGER.exception("History list failed")
            return jsonify({"status": "error", "message": "Analysis history is unavailable."}), 500

    @app.route("/api/history/<int:analysis_id>", methods=["GET"])
    def analysis_details(analysis_id):
        try:
            analysis = get_analysis_by_id(analysis_id)
            if analysis is None:
                return jsonify({"status": "error", "message": "Analysis not found."}), 404
            return jsonify({"status": "success", "analysis": analysis})
        except Exception:
            LOGGER.exception("History details failed")
            return jsonify({"status": "error", "message": "Analysis history is unavailable."}), 500

    @app.route("/api/history", methods=["DELETE"])
    def clear_history():
        try:
            clear_all_analyses()
            return jsonify({"status": "success", "message": "All analysis history cleared."}), 200
        except Exception:
            LOGGER.exception("Clear history failed")
            return jsonify({"status": "error", "message": "Failed to clear analysis history."}), 500

    @app.route("/api/history/<int:analysis_id>", methods=["DELETE"])
    def delete_single_history(analysis_id):
        try:
            deleted = delete_analysis_by_id(analysis_id)
            if not deleted:
                return jsonify({"status": "error", "message": "Analysis not found."}), 404
            return jsonify({"status": "success", "message": f"Analysis #{analysis_id} deleted."}), 200
        except Exception:
            LOGGER.exception("Delete history failed")
            return jsonify({"status": "error", "message": "Failed to delete analysis."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
