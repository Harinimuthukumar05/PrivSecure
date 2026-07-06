"""
app.py — PrivSecure Integration Server
=======================================
Bridges PrivSecureUI.html ↔ data_redaction_testing.py ↔ storage_backend.py

USAGE:
    python app.py

Then open:
    http://localhost:5000

INSTALL DEPENDENCIES:
    pip install flask flask-cors
    (plus all deps listed in data_redaction_testing.py and storage_backend.py)
"""

import os
import json
import tempfile
import logging
from flask_cors import CORS
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Core OCR/redaction pipeline ──────────────────────────────────────────────
from data_redaction_testing import extract_id_data, redact_sensitive_fields, ID_FIELDS

# ── Storage & verification backend ───────────────────────────────────────────
from storage_backend import (
    generate_verification_id,
    generate_access_key,
    store_result,
    retrieve_result,
    verify_document,
    get_blockchain_status,
    safe_json_serialize,
)

# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "pdf"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the frontend UI."""
    return send_from_directory(".", "PrivSecureUI.html")


# ── /api/extract ─────────────────────────────────────────────────────────────

@app.route("/api/extract", methods=["POST"])
def api_extract():
    """
    POST /api/extract
    -----------------
    Accepts a multipart file upload, runs the full pipeline:
      1. OCR extraction
      2. Verification ID + Access Key generation
      3. Storage (Firebase → local fallback + immutable ledger)

    Response shape:
    {
        "success": true,
        "id_type": "Aadhaar" | "PAN" | ...,
        "extracted": { ...all fields (unredacted)... },
        "redacted":  { ...sensitive fields masked... },
        "verification_id": "VERIFY-AB12CD",
        "access_key":      "ACCESS-XYZ123ABC",
        "confidence": 85,
        "blockchain_verified": true | false,
        "storage_status": "firebase+blockchain" | "local" | ...,
        "verification_status": "VERIFIED" | "PARTIAL",
        "status_messages": ["step 1", ...]
    }

    SECURITY NOTE:
        - verification_id is safe to share publicly (used for authenticity checks)
        - access_key must be kept private (required to retrieve stored data)
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS).upper()}"
        }), 415

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        print("EXTRACT API HIT")
        logger.info(f"Processing upload: {file.filename} → {tmp_path}")
        status_messages = ["File received and saved"]

        # ── Step 1: OCR extraction ────────────────────────────────────────
        extracted = extract_id_data(tmp_path)
        extracted["source_file"] = file.filename
        status_messages += [
            "Image preprocessing complete",
            "PaddleOCR multi-variant pass complete",
            "OCR lines deduplicated and cleaned",
        ]

        id_type = extracted.get("id_type", "Other")
        status_messages.append(f"ID type detected: {id_type}")
        status_messages += [
            "Regex extraction complete",
            "LLM structured extraction complete",
            "Field validation and filtering complete",
        ]

        # ── Step 2: Confidence ────────────────────────────────────────────
        fields = ID_FIELDS.get(id_type, ID_FIELDS["Other"])
        filled = sum(1 for f in fields if extracted.get(f) not in (None, "", "null"))
        confidence = round((filled / max(len(fields), 1)) * 100)

        # ── Step 3: Redaction ─────────────────────────────────────────────
        redacted = redact_sensitive_fields(extracted, mode="partial")
        status_messages.append("Redaction applied successfully")

        # ── Step 4: Generate public verification ID + private access key ──
        verification_id = generate_verification_id()
        access_key      = generate_access_key()
        status_messages.append(f"Verification ID generated: {verification_id}")
        status_messages.append("Private access key generated (shown once — keep it secure)")

        # ── Step 5: Storage (Firebase + local + ledger) ───────────────────
        storage_status_detail = store_result(
            verification_id=verification_id,
            access_key=access_key,
            result=extracted,        # store unredacted; redact at display layer
            document_name=file.filename,
        )

        backends_ok = []
        if storage_status_detail["firebase"]:
            backends_ok.append("firebase")
        if storage_status_detail["local"]:
            backends_ok.append("local")
        if storage_status_detail["blockchain"]:
            backends_ok.append("blockchain")
        storage_status_str  = "+".join(backends_ok) if backends_ok else "none"
        blockchain_verified = storage_status_detail["blockchain"]

        if blockchain_verified:
            status_messages.append(
                f"Immutable ledger block appended · "
                f"chain_hash={storage_status_detail.get('blockchain_tx', '')[:16]}…"
            )
        else:
            bc_err = storage_status_detail.get("blockchain_error") or "ledger write failed"
            status_messages.append(f"Ledger write error ({bc_err}) — data stored locally")

        verification_status = "VERIFIED" if blockchain_verified else "PARTIAL"
        status_messages.append(f"Verification status: {verification_status}")

        return jsonify({
            "success":            True,
            "id_type":            id_type,
            "extracted":          extracted,
            "redacted":           redacted,
            "verification_id":    verification_id,
            "access_key":         access_key,           # shown once — user must save it
            "confidence":         confidence,
            "blockchain_verified": blockchain_verified,
            "blockchain_tx":      storage_status_detail.get("blockchain_tx"),
            "storage_status":     storage_status_str,
            "verification_status": verification_status,
            "status_messages":    status_messages,
        })

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return jsonify({"success": False, "error": str(e)}), 404

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Processing failed: {str(e)}"}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── /api/verify ──────────────────────────────────────────────────────────────

@app.route("/api/verify", methods=["POST"])
def api_verify():
    """
    POST /api/verify
    ----------------
    Body (JSON): { "verification_id": "VERIFY-AB12CD" }

    Performs tamper-detection verification using ONLY the public verification_id.
    NEVER returns stored document data.

    Response shape:
    {
        "success": true,
        "verification_id": "VERIFY-AB12CD",
        "verified": true | false,
        "tamper_detected": false,
        "blockchain_verified": true,
        "chain_intact": true,
        "data_found": true,
        "blockchain_record": { block_index, timestamp, document_name, ... },
        "verification_status": "VERIFIED" | "TAMPER_DETECTED" | "PARTIAL" | "NOT_FOUND",
        "timestamp": 1234567890
    }

    SECURITY: no document data is exposed in this response.
    """
    body = request.get_json(silent=True) or {}
    verification_id = (body.get("verification_id") or "").strip().upper()

    if not verification_id:
        return jsonify({"success": False, "error": "verification_id is required."}), 400

    try:
        verification = verify_document(verification_id)

        return jsonify({
            "success":         True,
            "verification_id": verification_id,
            **verification,
            # Explicitly exclude any data field — verification must not leak data
        })

    except Exception as e:
        logger.exception(f"Verification error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── /api/retrieve ─────────────────────────────────────────────────────────────

@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    """
    POST /api/retrieve
    ------------------
    Body (JSON): {
        "verification_id": "VERIFY-AB12CD",
        "access_key":      "ACCESS-XYZ123ABC"
    }

    Both fields are required. verification_id alone will NOT return data.
    If access_key is wrong, returns 401 Unauthorized.

    Response shape:
    {
        "success": true,
        "found": true,
        "verification_id": "VERIFY-AB12CD",
        "data": { ...extracted fields... },
        "stored_on": "firebase" | "local",
        "timestamp": 1234567890,
        "blockchain_verified": true,
        "blockchain_record": { block_index, timestamp, document_name,
                               previous_hash, current_hash, exists },
        "verification_status": "VERIFIED" | "PARTIAL" | "NOT_FOUND"
    }
    """
    body = request.get_json(silent=True) or {}
    verification_id = (body.get("verification_id") or "").strip().upper()
    access_key      = (body.get("access_key") or "").strip().upper()

    if not verification_id:
        return jsonify({"success": False, "error": "verification_id is required."}), 400
    if not access_key:
        return jsonify({"success": False, "error": "access_key is required."}), 400

    try:
        result = retrieve_result(verification_id, access_key)

        if result is None:
            return jsonify({
                "success":         True,
                "found":           False,
                "verification_id": verification_id,
            })

        # Ledger lookup for display
        from storage_backend import verify_hash_on_blockchain, get_blockchain_record
        bc_verified = verify_hash_on_blockchain(verification_id)
        bc_record   = get_blockchain_record(verification_id) if bc_verified else {"exists": False}

        verification_status = (
            "VERIFIED" if bc_verified and result else
            "PARTIAL"  if result else
            "NOT_FOUND"
        )

        return jsonify({
            "success":             True,
            "found":               True,
            "verification_id":     verification_id,
            "data":                result["data"],
            "stored_on":           result["stored_on"],
            "timestamp":           result["timestamp"],
            "blockchain_verified": bc_verified,
            "blockchain_record":   bc_record,
            "verification_status": verification_status,
        })

    except PermissionError:
        logger.warning(f"Unauthorized retrieval attempt for [{verification_id}]")
        return jsonify({
            "success": False,
            "error":   "Unauthorized Access — Invalid access key.",
            "unauthorized": True,
        }), 401

    except Exception as e:
        logger.exception(f"Retrieval error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── /api/status ───────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    """
    GET /api/status
    ---------------
    Health check: returns backend connectivity info (no sensitive data).
    """
    from storage_backend import firebase_enabled
    return jsonify({
        "server":           "PrivSecure Integration Server",
        "firebase_enabled": firebase_enabled,
        "blockchain":       get_blockchain_status(),
    })


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  PrivSecure Integration Server")
    print("  Open: http://localhost:5000")
    print("=" * 55 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)