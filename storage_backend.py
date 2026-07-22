# storage_backend.py
# ============================================================
# Storage & Verification Backend — PrivSecure
# ============================================================
# Architecture:
#   - Firebase Realtime Database (primary storage)
#   - Local JSON fallback storage  (always-on safety net)
#   - Blockchain-inspired immutable ledger (chained SHA-256)
#
# NEW in this version:
#   - Real tamper detection via stored original_data_hash
#   - Separate public verification_id (VERIFY-XXXXXX) and
#     private access_key (ACCESS-XXXXXXXXX) per document
#   - retrieve_result() requires BOTH verification_id + access_key
#   - verify_document() uses ONLY verification_id — never exposes data
#   - Immutable ledger blocks now include original_data_hash
# ============================================================

import os
import json
import time
import hashlib
import secrets
import uuid
import pathlib
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# Firebase Setup
# ============================================================

SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
DATABASE_URL = "https://data-redaction-default-rtdb.asia-southeast1.firebasedatabase.app/"

firebase_enabled = False
fb_db = None

try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db_module

    firebase_credentials_env = os.getenv("FIREBASE_CREDENTIALS")

    cred = None
    if firebase_credentials_env:
        try:
            parsed_json = json.loads(firebase_credentials_env.strip())
            cred = credentials.Certificate(parsed_json)
        except Exception as e:
            logger.error(f"❌ Failed to parse FIREBASE_CREDENTIALS: {e}")
    elif pathlib.Path(SERVICE_ACCOUNT_PATH).exists():
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    else:
        logger.warning("⚠️ serviceAccountKey.json not found — using local JSON storage.")

    if cred is not None:
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
            fb_db = firebase_db_module.reference("/")
            firebase_enabled = True
            logger.info("✅ Firebase connected")
        except Exception as e:
            logger.error(f"❌ Firebase init failed: {e}")
except ImportError:
    logger.warning("⚠️ firebase_admin not installed — using local JSON storage.")


# ============================================================
# Blockchain-Inspired Immutable Ledger
# ============================================================
# Each block now contains:
#   - block_index        : sequential position in the chain
#   - verification_id    : public identifier (VERIFY-XXXXXX)
#   - document_name      : original filename
#   - timestamp          : Unix epoch (UTC)
#   - original_data_hash : SHA-256 of sorted JSON of stored data
#                          (used for tamper detection on verify)
#   - previous_hash      : current_hash of preceding block
#                          (genesis block uses "0" * 64)
#   - current_hash       : SHA-256(verification_id + previous_hash)
#
# NOTE: access_key is intentionally NOT stored in the ledger —
# the ledger is public/auditable; the access_key must stay private.
#
# STORAGE:
# The ledger is stored in Firebase Realtime Database under:
#   ledger/
#       block_0
#       block_1
#       block_2
#       ...
# This replaces the previous local blockchain_ledger.json file, which
# did not survive Railway's ephemeral filesystem across deployments.
# ============================================================

LEDGER_ROOT = "ledger"
_GENESIS_PREVIOUS_HASH = "0" * 64


def _load_ledger() -> list[dict]:
    """Load the full ledger from Firebase. Returns [] if not yet created."""
    if not firebase_enabled or fb_db is None:
        logger.error("❌ Firebase not available — cannot load ledger.")
        return []
    try:
        raw = fb_db.child(LEDGER_ROOT).get()
        if not raw or not isinstance(raw, dict):
            return []
        blocks = [block for block in raw.values() if isinstance(block, dict)]
        blocks.sort(key=lambda b: b.get("block_index", 0))
        return blocks
    except Exception as e:
        logger.error(f"❌ Could not read ledger from Firebase: {e}")
        return []


def _save_ledger(chain: list[dict]) -> bool:
    """Persist the full ledger to Firebase under ledger/block_<index>."""
    if not firebase_enabled or fb_db is None:
        logger.error("❌ Firebase not available — cannot write ledger.")
        return False
    try:
        ledger_data = {
            f"block_{block['block_index']}": block for block in chain
        }
        fb_db.child(LEDGER_ROOT).set(ledger_data)
        return True
    except Exception as e:
        logger.error(f"❌ Could not write ledger to Firebase: {e}")
        return False


def _compute_block_hash(verification_id: str, previous_hash: str) -> str:
    """
    Deterministic SHA-256 digest for a single ledger block.
    current_hash = SHA-256(verification_id + previous_hash)
    """
    payload = (verification_id + previous_hash).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def store_hash_on_blockchain(
    verification_id: str,
    document_name: str,
    original_data_hash: str,
) -> dict:
    """
    Append a new block to the blockchain-inspired immutable ledger.

    Args:
        verification_id:    Public VERIFY-XXXXXX identifier.
        document_name:      Original filename for audit trail.
        original_data_hash: SHA-256 of the stored document data
                            (enables tamper detection during verify).

    Returns:
        {
            "success":  bool,
            "tx_hash":  str | None,   # current_hash of the new block
            "error":    str | None,
        }
    """
    try:
        chain = _load_ledger()

        previous_hash = (
            chain[-1]["current_hash"] if chain else _GENESIS_PREVIOUS_HASH
        )
        current_hash = _compute_block_hash(verification_id, previous_hash)

        new_block = {
            "block_index":        len(chain),
            "verification_id":    verification_id,
            "document_name":      document_name,
            "timestamp":          int(time.time()),
            "original_data_hash": original_data_hash,   # ← tamper-detection anchor
            "previous_hash":      previous_hash,
            "current_hash":       current_hash,
        }

        chain.append(new_block)

        if not _save_ledger(chain):
            return {"success": False, "tx_hash": None, "error": "Failed to persist ledger."}

        logger.info(
            f"✅ Ledger block #{new_block['block_index']} added "
            f"[vid={verification_id} | data_hash={original_data_hash[:16]}…]"
        )
        return {"success": True, "tx_hash": current_hash, "error": None}

    except Exception as e:
        logger.error(f"❌ Ledger append failed: {e}")
        return {"success": False, "tx_hash": None, "error": str(e)}


def verify_hash_on_blockchain(verification_id: str) -> bool:
    """Check whether verification_id is recorded in the ledger."""
    chain = _load_ledger()
    return any(block.get("verification_id") == verification_id for block in chain)


def get_blockchain_record(verification_id: str) -> dict:
    """
    Retrieve the ledger block for a given verification_id.
    Returns full block dict + "exists": True, or {"exists": False}.
    """
    chain = _load_ledger()
    for block in chain:
        if block.get("verification_id") == verification_id:
            return {**block, "exists": True}
    return {"exists": False}


def get_blockchain_status() -> dict:
    """Return a health summary of the immutable ledger."""
    chain = _load_ledger()
    return {
        "type":         "blockchain-inspired immutable ledger",
        "ledger_file":  f"firebase:/{LEDGER_ROOT}",
        "block_count":  len(chain),
        "chain_intact": _verify_chain_integrity(chain),
        "connected":    True,
    }


def _verify_chain_integrity(chain: list[dict]) -> bool:
    """
    Walk the ledger and verify every block's current_hash is consistent
    with its verification_id and previous_hash.
    Returns True if the chain is intact.
    """
    if not chain:
        return True
    for i, block in enumerate(chain):
        expected = _compute_block_hash(
            block.get("verification_id", ""),
            block.get("previous_hash", ""),
        )
        if block.get("current_hash") != expected:
            logger.warning(f"⚠️ Chain integrity violation at block #{i}")
            return False
        if i > 0 and block["previous_hash"] != chain[i - 1]["current_hash"]:
            logger.warning(f"⚠️ Chain link broken between blocks #{i-1} and #{i}")
            return False
    return True


# ============================================================
# ID Generation
# ============================================================

def generate_verification_id() -> str:
    """
    Generate a short, URL-safe public verification identifier.
    Format: VERIFY-XXXXXX (6 uppercase hex chars).
    Safe to share publicly — used only for authenticity checks.
    """
    return "VERIFY-" + secrets.token_hex(3).upper()


def generate_access_key() -> str:
    """
    Generate a cryptographically secure private access key.
    Format: ACCESS-XXXXXXXXX (9 uppercase alphanumeric chars).
    Must remain secret — required to retrieve document data.
    """
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # unambiguous chars
    token = "".join(secrets.choice(alphabet) for _ in range(9))
    return "ACCESS-" + token


def compute_data_hash(result: dict) -> str:
    """
    Compute SHA-256 of a document result dict (sorted keys).
    This hash is stored at upload time and re-computed at verify time
    to detect any tampering with the stored data.
    """
    return hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# ============================================================
# Firebase Functions
# ============================================================

def save_to_firebase(verification_id: str, payload: dict) -> bool:
    """Save the full storage payload to Firebase under verification_id."""
    if not firebase_enabled or fb_db is None:
        return False
    try:
        fb_db.child(f"id_extractions/{verification_id}").set(payload)
        return True
    except Exception as e:
        logger.error(f"❌ Firebase write failed: {e}")
        return False


def get_from_firebase(verification_id: str) -> dict | None:
    """Retrieve the full storage payload from Firebase."""
    if not firebase_enabled or fb_db is None:
        return None
    try:
        return fb_db.child(f"id_extractions/{verification_id}").get()
    except Exception as e:
        logger.error(f"Firebase lookup error: {e}")
        return None


# ============================================================
# Local JSON Fallback Storage
# ============================================================

LOCAL_RESULTS_FILE = "local_results.json"


def save_to_local(verification_id: str, payload: dict) -> bool:
    """Save the full storage payload to local JSON file."""
    try:
        local_data: dict = {}
        if os.path.exists(LOCAL_RESULTS_FILE):
            with open(LOCAL_RESULTS_FILE, "r", encoding="utf-8") as f:
                try:
                    local_data = json.load(f)
                except json.JSONDecodeError:
                    local_data = {}
        local_data[verification_id] = payload
        with open(LOCAL_RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write(safe_json_serialize(local_data))
        return True
    except Exception as e:
        logger.error(f"❌ Local save failed: {e}")
        return False


def get_from_local(verification_id: str) -> dict | None:
    """Retrieve the full storage payload from local JSON file."""
    try:
        if os.path.exists(LOCAL_RESULTS_FILE):
            with open(LOCAL_RESULTS_FILE, "r", encoding="utf-8") as f:
                local_data = json.load(f)
            return local_data.get(verification_id)
    except Exception as e:
        logger.error(f"Local lookup error: {e}")
    return None


# ============================================================
# Unified Storage Functions
# ============================================================

def store_result(
    verification_id: str,
    access_key: str,
    result: dict,
    document_name: str,
) -> dict:
    """
    Persist extraction result across all available backends.

    Storage payload structure:
    {
        "access_key":         str,             # hashed — never stored in plain text
        "original_data_hash": str,             # SHA-256 of result for tamper detection
        "data":               dict,            # the extracted document data
        "document_name":      str,
        "timestamp":          int,
    }

    NOTE: access_key is stored as a SHA-256 hash, never in plain text.
    The ledger block stores original_data_hash but NOT access_key.

    Returns a status summary dict.
    """
    status = {
        "firebase":         False,
        "local":            False,
        "blockchain":       False,
        "blockchain_tx":    None,
        "blockchain_error": None,
    }

    # Compute the original data hash for tamper detection
    original_data_hash = compute_data_hash(result)

    # Hash the access key for secure storage (one-way)
    access_key_hash = hashlib.sha256(access_key.encode()).hexdigest()

    payload = {
        "access_key_hash":    access_key_hash,
        "original_data_hash": original_data_hash,
        "data":               result,
        "document_name":      document_name,
        "timestamp":          int(time.time()),
    }

    # 1. Firebase
    if firebase_enabled:
        status["firebase"] = save_to_firebase(verification_id, payload)
        if status["firebase"]:
            logger.info(f"✅ Saved to Firebase [{verification_id}]")
        else:
            logger.warning(f"⚠️ Firebase save failed [{verification_id}]")

    # 2. Local fallback (always run — data survives Firebase outages)
    status["local"] = save_to_local(verification_id, payload)
    if status["local"]:
        logger.info(f"✅ Saved to local storage [{verification_id}]")

    # 3. Immutable ledger — stores verification_id + original_data_hash, NOT access_key
    ledger_result = store_hash_on_blockchain(
        verification_id=verification_id,
        document_name=document_name,
        original_data_hash=original_data_hash,
    )
    status["blockchain"]       = ledger_result["success"]
    status["blockchain_tx"]    = ledger_result.get("tx_hash")
    status["blockchain_error"] = ledger_result.get("error")

    if status["blockchain"]:
        logger.info(f"✅ Hash anchored in immutable ledger [{verification_id}]")
    else:
        logger.warning(f"⚠️ Ledger write failed: {ledger_result.get('error')}")

    return status


def retrieve_result(verification_id: str, access_key: str) -> dict | None:
    """
    Retrieve extraction data.  Requires BOTH verification_id AND access_key.

    - verification_id alone is NOT sufficient — returns None (unauthorized).
    - access_key is compared against its stored SHA-256 hash.

    Returns:
        dict with keys: "data", "document_name", "timestamp", "stored_on"
        None if not found or access_key is wrong.

    Raises:
        PermissionError if the record exists but access_key is incorrect.
    """
    payload = None
    stored_on = None

    if firebase_enabled:
        payload = get_from_firebase(verification_id)
        if payload:
            stored_on = "firebase"

    if payload is None:
        payload = get_from_local(verification_id)
        if payload:
            stored_on = "local"

    if payload is None:
        return None  # record not found

    # Validate access key
    provided_hash = hashlib.sha256(access_key.encode()).hexdigest()
    stored_hash   = payload.get("access_key_hash", "")

    if provided_hash != stored_hash:
        logger.warning(f"⚠️ Unauthorized retrieval attempt for [{verification_id}]")
        raise PermissionError("Invalid access key.")

    logger.info(f"✅ Retrieved from {stored_on} [{verification_id}]")
    return {
        "data":          payload.get("data", {}),
        "document_name": payload.get("document_name", ""),
        "timestamp":     payload.get("timestamp"),
        "stored_on":     stored_on,
    }


def verify_document(verification_id: str) -> dict:
    """
    Full tamper-detection verification flow using ONLY verification_id.

    Steps:
    1. Check chain integrity of the full ledger.
    2. Look up the ledger block for verification_id.
    3. Retrieve stored data (without requiring access_key — just for hash comparison).
    4. Re-compute current_data_hash and compare with original_data_hash from ledger.

    Verification Rules:
        VERIFIED        — ledger record exists, chain intact, data hashes match
        TAMPER_DETECTED — ledger record exists, but stored data has been modified
        PARTIAL         — data exists in storage but no ledger record
        NOT_FOUND       — no record found anywhere

    Returns:
    {
        "verified":            bool,
        "tamper_detected":     bool,
        "blockchain_verified": bool,
        "data_found":          bool,
        "chain_intact":        bool,
        "blockchain_record":   dict,
        "verification_status": "VERIFIED" | "TAMPER_DETECTED" | "PARTIAL" | "NOT_FOUND",
        "timestamp":           int | None,
    }

    NOTE: This function intentionally does NOT return any document data —
    only verification metadata is exposed.
    """
    # 1. Validate full chain integrity first
    chain = _load_ledger()
    chain_intact = _verify_chain_integrity(chain)

    # 2. Look for this verification_id in the ledger
    ledger_record  = get_blockchain_record(verification_id)
    ledger_verified = ledger_record.get("exists", False)

    # 3. Retrieve payload for hash comparison (no access_key needed here —
    #    we only need to re-hash the stored data, not return it to caller)
    payload = None
    if firebase_enabled:
        payload = get_from_firebase(verification_id)
    if payload is None:
        payload = get_from_local(verification_id)

    data_found = payload is not None

    # 4. Tamper detection
    tamper_detected = False
    if ledger_verified and data_found:
        stored_data          = payload.get("data", {})
        current_data_hash    = compute_data_hash(stored_data)
        original_data_hash   = ledger_record.get("original_data_hash", "")

        if original_data_hash and current_data_hash != original_data_hash:
            tamper_detected = True
            logger.warning(
                f"🚨 TAMPER DETECTED for [{verification_id}]: "
                f"stored_hash={current_data_hash[:16]}… "
                f"original_hash={original_data_hash[:16]}…"
            )

    # 5. Determine final status
    if not data_found and not ledger_verified:
        verification_status = "NOT_FOUND"
        verified = False
    elif data_found and not ledger_verified:
        verification_status = "PARTIAL"
        verified = False
    elif tamper_detected:
        verification_status = "TAMPER_DETECTED"
        verified = False
    elif ledger_verified and data_found and chain_intact:
        verification_status = "VERIFIED"
        verified = True
    else:
        # Ledger record exists but chain is broken or other partial state
        verification_status = "PARTIAL"
        verified = False

    timestamp = ledger_record.get("timestamp") if ledger_verified else None

    return {
        "verified":            verified,
        "tamper_detected":     tamper_detected,
        "blockchain_verified": ledger_verified,
        "data_found":          data_found,
        "chain_intact":        chain_intact,
        "blockchain_record":   ledger_record,
        "verification_status": verification_status,
        "timestamp":           timestamp,
    }


# ============================================================
# Legacy hash_key helpers — kept for backward compatibility
# (old records stored under raw hash keys will still be found)
# ============================================================

def generate_hash_key(data: dict) -> str:
    """
    Legacy: Generate a short unique SHA-256 hash key.
    New code should use generate_verification_id() + generate_access_key().
    Kept for backward-compatibility with any pre-existing records.
    """
    unique_data = {"data": data, "uuid": str(uuid.uuid4())}
    raw = json.dumps(unique_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


# ============================================================
# Secure Helper Functions
# ============================================================

def safe_json_serialize(obj) -> str:
    """JSON serialize with a graceful fallback for non-serialisable objects."""
    def default_handler(o):
        if isinstance(o, bytes):
            return o.hex()
        return str(o)
    return json.dumps(obj, indent=4, ensure_ascii=False, default=default_handler)