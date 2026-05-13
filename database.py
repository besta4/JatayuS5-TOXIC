"""
database.py — SQLite persistence layer for Jatayu.

Tables (Legacy - Batch Processing):
  tasks               — upload metadata + status
  transaction_results — one row per processed transaction (JSON blob)
  audit_records       — agent 5 compliance log entries (JSON blob)

Tables (Real-Time System):
  users               — user accounts (CUSTOMER, MERCHANT, ADMIN)
  user_profiles       — extended user profile information
  accounts            — financial accounts with balances
  sessions            — JWT sessions for auth
  device_registry     — trusted device tracking
  transactions        — real-time transactions with fraud pipeline results
  payees              — saved payees
  velocity_tracking   — transaction velocity per user
  compliance_reports  — STR/CTR compliance reports
  pending_otp         — pending OTP verifications for email-protected transactions
"""

from __future__ import annotations

import json
import sqlite3
import uuid
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from enum import Enum

DB_PATH = Path(__file__).parent / "jatayu.db"


# ── Enums ────────────────────────────────────────────────────────────────────

class UserType(str, Enum):
    CUSTOMER = "CUSTOMER"
    MERCHANT = "MERCHANT"
    ADMIN = "ADMIN"


class KYCLevel(str, Enum):
    """Legacy KYC levels - kept for compatibility but not used."""
    MINIMUM = "MINIMUM"
    ENHANCED = "ENHANCED"
    FULL = "FULL"


class AccountStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    BLOCKED = "BLOCKED"


class AccountType(str, Enum):
    SAVINGS = "SAVINGS"
    CURRENT = "CURRENT"
    MERCHANT = "MERCHANT"
    ESCROW = "ESCROW"


class TransactionStatus(str, Enum):
    INITIATED = "INITIATED"
    PENDING_FRAUD = "PENDING_FRAUD"
    HELD = "HELD"
    APPROVED = "APPROVED"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    REVERSED = "REVERSED"


class TransactionType(str, Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"


# ── Connection Manager ───────────────────────────────────────────────────────

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize all database tables."""
    with get_conn() as conn:
        conn.executescript("""
            -- ================================================================
            -- LEGACY TABLES (Batch Processing)
            -- ================================================================

            CREATE TABLE IF NOT EXISTS tasks (
                task_id      TEXT PRIMARY KEY,
                filename     TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                total_rows   INTEGER DEFAULT 0,
                processed    INTEGER DEFAULT 0,
                fraud_count  INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS transaction_results (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id        TEXT NOT NULL,
                transaction_id TEXT NOT NULL,
                data           TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id)
            );

            CREATE TABLE IF NOT EXISTS audit_records (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id    TEXT NOT NULL,
                record     TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id)
            );

            -- ================================================================
            -- USERS & AUTHENTICATION
            -- ================================================================

            CREATE TABLE IF NOT EXISTS users (
                user_id         TEXT PRIMARY KEY,
                email           TEXT UNIQUE NOT NULL,
                phone           TEXT UNIQUE,
                password_hash   TEXT NOT NULL,
                user_type       TEXT NOT NULL CHECK (user_type IN ('CUSTOMER', 'MERCHANT', 'ADMIN')),
                account_status  TEXT DEFAULT 'ACTIVE' CHECK (account_status IN ('PENDING', 'ACTIVE', 'SUSPENDED', 'BLOCKED')),
                email_otp_enabled INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                last_login      TEXT,
                failed_attempts INTEGER DEFAULT 0,
                locked_until    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_users_type ON users(user_type);

            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id         TEXT PRIMARY KEY,
                display_name    TEXT NOT NULL,
                business_name   TEXT,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            -- ================================================================
            -- ACCOUNTS & BALANCES
            -- ================================================================

            CREATE TABLE IF NOT EXISTS accounts (
                account_id      TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                account_type    TEXT NOT NULL CHECK (account_type IN ('SAVINGS', 'CURRENT', 'MERCHANT', 'ESCROW')),
                balance         REAL NOT NULL DEFAULT 0.0 CHECK (balance >= 0),
                currency        TEXT DEFAULT 'INR',
                daily_limit     REAL,
                is_primary      INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_accounts_user ON accounts(user_id);

            -- ================================================================
            -- SESSIONS & DEVICE TRACKING
            -- ================================================================

            CREATE TABLE IF NOT EXISTS sessions (
                session_id      TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                token_hash      TEXT NOT NULL,
                device_id       TEXT NOT NULL,
                ip_address      TEXT NOT NULL,
                user_agent      TEXT,
                created_at      TEXT NOT NULL,
                expires_at      TEXT NOT NULL,
                is_active       INTEGER DEFAULT 1,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash);

            CREATE TABLE IF NOT EXISTS device_registry (
                device_id       TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                device_name     TEXT,
                device_type     TEXT CHECK (device_type IN ('MOBILE', 'WEB', 'DESKTOP')),
                is_trusted      INTEGER DEFAULT 0,
                first_seen      TEXT NOT NULL,
                last_seen       TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_devices_user ON device_registry(user_id);

            -- ================================================================
            -- REAL-TIME TRANSACTIONS
            -- ================================================================

            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id  TEXT PRIMARY KEY,
                step            INTEGER NOT NULL,
                type            TEXT NOT NULL CHECK (type IN ('PAYMENT', 'TRANSFER', 'CASH_IN', 'CASH_OUT', 'DEBIT')),
                amount          REAL NOT NULL CHECK (amount > 0),

                sender_id       TEXT NOT NULL,
                sender_account  TEXT NOT NULL,
                receiver_id     TEXT NOT NULL,
                receiver_account TEXT NOT NULL,

                old_balance_sender   REAL NOT NULL,
                new_balance_sender   REAL NOT NULL,
                old_balance_receiver REAL NOT NULL,
                new_balance_receiver REAL NOT NULL,

                ip_address      TEXT NOT NULL,
                device_id       TEXT NOT NULL,
                description     TEXT,
                reference_id    TEXT,

                status          TEXT NOT NULL DEFAULT 'INITIATED' CHECK (status IN (
                    'INITIATED', 'PENDING_FRAUD', 'HELD', 'APPROVED',
                    'COMPLETED', 'BLOCKED', 'FAILED', 'REVERSED'
                )),

                initiated_at    TEXT NOT NULL,
                fraud_check_at  TEXT,
                completed_at    TEXT,

                fraud_score         REAL,
                fraud_label         INTEGER,
                pattern_type        TEXT,
                pattern_confidence  REAL,
                risk_level          TEXT,
                recommended_action  TEXT,
                action_taken        TEXT,
                explanation         TEXT,
                pipeline_latency_ms REAL,

                FOREIGN KEY(sender_id) REFERENCES users(user_id),
                FOREIGN KEY(receiver_id) REFERENCES users(user_id),
                FOREIGN KEY(sender_account) REFERENCES accounts(account_id),
                FOREIGN KEY(receiver_account) REFERENCES accounts(account_id)
            );

            CREATE INDEX IF NOT EXISTS idx_txn_sender ON transactions(sender_id, initiated_at);
            CREATE INDEX IF NOT EXISTS idx_txn_receiver ON transactions(receiver_id, initiated_at);
            CREATE INDEX IF NOT EXISTS idx_txn_status ON transactions(status);
            CREATE INDEX IF NOT EXISTS idx_txn_fraud ON transactions(fraud_label, risk_level);

            -- ================================================================
            -- PAYEE MANAGEMENT
            -- ================================================================

            CREATE TABLE IF NOT EXISTS payees (
                payee_id        TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                payee_user_id   TEXT NOT NULL,
                nickname        TEXT,
                added_at        TEXT NOT NULL,
                is_verified     INTEGER DEFAULT 0,
                total_txn_count INTEGER DEFAULT 0,
                total_txn_amount REAL DEFAULT 0.0,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY(payee_user_id) REFERENCES users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_payees_user ON payees(user_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_payees_unique ON payees(user_id, payee_user_id);

            -- ================================================================
            -- COMPLIANCE & VELOCITY TRACKING
            -- ================================================================

            CREATE TABLE IF NOT EXISTS velocity_tracking (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT NOT NULL,
                window_type     TEXT NOT NULL CHECK (window_type IN ('HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY')),
                window_start    TEXT NOT NULL,
                txn_count       INTEGER DEFAULT 0,
                txn_amount      REAL DEFAULT 0.0,
                updated_at      TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_velocity_user ON velocity_tracking(user_id, window_type, window_start);

            CREATE TABLE IF NOT EXISTS compliance_reports (
                report_id       TEXT PRIMARY KEY,
                report_type     TEXT NOT NULL CHECK (report_type IN ('STR', 'CTR', 'OFAC', 'DAILY_SUMMARY')),
                transaction_id  TEXT,
                user_id         TEXT,
                amount          REAL,
                trigger_reason  TEXT NOT NULL,
                auto_generated  INTEGER DEFAULT 1,
                submitted_at    TEXT,
                reviewed_by     TEXT,
                status          TEXT DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'SUBMITTED', 'ACKNOWLEDGED')),
                created_at      TEXT NOT NULL,
                FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id),
                FOREIGN KEY(user_id) REFERENCES users(user_id),
                FOREIGN KEY(reviewed_by) REFERENCES users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_compliance_type ON compliance_reports(report_type, status);

            -- ================================================================
            -- EMAIL OTP FOR PAYMENTS
            -- ================================================================

            CREATE TABLE IF NOT EXISTS pending_otp (
                otp_id          TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                otp_hash        TEXT NOT NULL,
                transaction_data TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                expires_at      TEXT NOT NULL,
                attempts        INTEGER DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_otp_user ON pending_otp(user_id);

            -- ================================================================
            -- SUPPORT TICKETS (for blocked/suspended users to contact admin)
            -- ================================================================

            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id       TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                subject         TEXT NOT NULL,
                status          TEXT DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'IN_PROGRESS', 'RESOLVED', 'CLOSED')),
                priority        TEXT DEFAULT 'NORMAL' CHECK (priority IN ('LOW', 'NORMAL', 'HIGH', 'URGENT')),
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                resolved_at     TEXT,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tickets_user ON support_tickets(user_id);
            CREATE INDEX IF NOT EXISTS idx_tickets_status ON support_tickets(status);

            CREATE TABLE IF NOT EXISTS support_messages (
                message_id      TEXT PRIMARY KEY,
                ticket_id       TEXT NOT NULL,
                sender_id       TEXT NOT NULL,
                sender_role     TEXT NOT NULL CHECK (sender_role IN ('USER', 'ADMIN')),
                message         TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                FOREIGN KEY(ticket_id) REFERENCES support_tickets(ticket_id) ON DELETE CASCADE,
                FOREIGN KEY(sender_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_ticket ON support_messages(ticket_id);
        """)

        # Migrations for existing databases
        try:
            conn.execute("SELECT reviewed_by FROM compliance_reports LIMIT 1")
        except Exception:
            conn.execute("ALTER TABLE compliance_reports ADD COLUMN reviewed_by TEXT")
        
        try:
            conn.execute("SELECT email_otp_enabled FROM users LIMIT 1")
        except Exception:
            conn.execute("ALTER TABLE users ADD COLUMN email_otp_enabled INTEGER DEFAULT 0")


# ── Task helpers ──────────────────────────────────────────────────────────────

def create_task(task_id: str, filename: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tasks (task_id, filename, status, created_at) VALUES (?, ?, 'pending', ?)",
            (task_id, filename, datetime.now(timezone.utc).isoformat()),
        )


def update_task(task_id: str, **kwargs) -> None:
    """Update any subset of task columns."""
    if not kwargs:
        return
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [task_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE tasks SET {sets} WHERE task_id = ?", vals)


def get_task(task_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


def list_tasks() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


# ── Transaction result helpers ────────────────────────────────────────────────

def save_transaction(task_id: str, transaction_id: str, data: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO transaction_results (task_id, transaction_id, data) VALUES (?, ?, ?)",
            (task_id, transaction_id, json.dumps(data, default=str)),
        )


def get_results(task_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT data FROM transaction_results WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    return [json.loads(r["data"]) for r in rows]


# ── Audit record helpers ──────────────────────────────────────────────────────

def save_audit_record(task_id: str, record: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_records (task_id, record, created_at) VALUES (?, ?, ?)",
            (task_id, json.dumps(record, default=str), datetime.now(timezone.utc).isoformat()),
        )


def get_audit_records(task_id: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if task_id:
            rows = conn.execute(
                "SELECT record, created_at FROM audit_records WHERE task_id = ? ORDER BY id",
                (task_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT record, created_at FROM audit_records ORDER BY id DESC LIMIT 500"
            ).fetchall()
    return [{"record": json.loads(r["record"]), "created_at": r["created_at"]} for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def generate_user_id(user_type: UserType) -> str:
    """Generate sequential user ID like C000000001, M000000001, A000000001."""
    prefix = user_type.value[0]
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM users WHERE user_type = ?",
            (user_type.value,)
        ).fetchone()
        count = row["cnt"] + 1
    return f"{prefix}{count:09d}"


def create_user(
    email: str,
    password_hash: str,
    user_type: UserType,
    display_name: str,
    phone: Optional[str] = None,
    business_name: Optional[str] = None
) -> str:
    """Create a new user and profile. Returns user_id."""
    user_id = generate_user_id(user_type)
    now = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO users
               (user_id, email, phone, password_hash, user_type, account_status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?)""",
            (user_id, email, phone, password_hash, user_type.value, now, now)
        )
        conn.execute(
            """INSERT INTO user_profiles (user_id, display_name, business_name)
               VALUES (?, ?, ?)""",
            (user_id, display_name, business_name)
        )
    return user_id


def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email for login."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user by ID."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_user_with_profile(user_id: str) -> Optional[dict]:
    """Get user with profile info."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT u.*, p.display_name, p.business_name
               FROM users u
               LEFT JOIN user_profiles p ON u.user_id = p.user_id
               WHERE u.user_id = ?""",
            (user_id,)
        ).fetchone()
    return dict(row) if row else None


def update_user(user_id: str, **kwargs) -> None:
    """Update user fields."""
    if not kwargs:
        return
    kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [user_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE users SET {sets} WHERE user_id = ?", vals)


def update_user_login(user_id: str) -> None:
    """Update last login timestamp and reset failed attempts."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = ?, failed_attempts = 0, locked_until = NULL WHERE user_id = ?",
            (now, user_id)
        )


def increment_failed_login(user_id: str) -> int:
    """Increment failed login attempts. Returns new count."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET failed_attempts = failed_attempts + 1 WHERE user_id = ?",
            (user_id,)
        )
        row = conn.execute(
            "SELECT failed_attempts FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["failed_attempts"] if row else 0


def lock_user(user_id: str, minutes: int = 30) -> None:
    """Lock user account for specified minutes."""
    locked_until = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET locked_until = ? WHERE user_id = ?",
            (locked_until, user_id)
        )


def list_users(user_type: Optional[UserType] = None, limit: int = 100) -> list[dict]:
    """List users, optionally filtered by type."""
    with get_conn() as conn:
        if user_type:
            rows = conn.execute(
                """SELECT u.*, p.display_name, p.business_name
                   FROM users u LEFT JOIN user_profiles p ON u.user_id = p.user_id
                   WHERE u.user_type = ? ORDER BY u.created_at DESC LIMIT ?""",
                (user_type.value, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT u.*, p.display_name, p.business_name
                   FROM users u LEFT JOIN user_profiles p ON u.user_id = p.user_id
                   ORDER BY u.created_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# ACCOUNT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def generate_account_id() -> str:
    """Generate unique account ID."""
    return f"ACC{uuid.uuid4().hex[:12].upper()}"


def create_account(
    user_id: str,
    account_type: AccountType,
    initial_balance: float = 0.0,
    is_primary: bool = False
) -> str:
    """Create a new account. Returns account_id."""
    account_id = generate_account_id()
    now = datetime.now(timezone.utc).isoformat()

    # Default daily limit (can be configured per account if needed)
    daily_limit = 200000  # ₹2 lakh default

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO accounts
               (account_id, user_id, account_type, balance, daily_limit, is_primary, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (account_id, user_id, account_type.value, initial_balance, daily_limit, 1 if is_primary else 0, now, now)
        )
    return account_id


def get_account(account_id: str) -> Optional[dict]:
    """Get account by ID."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
        ).fetchone()
    return dict(row) if row else None


def get_user_accounts(user_id: str) -> list[dict]:
    """Get all accounts for a user."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM accounts WHERE user_id = ? ORDER BY is_primary DESC",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_primary_account(user_id: str) -> Optional[dict]:
    """Get user's primary account."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE user_id = ? AND is_primary = 1",
            (user_id,)
        ).fetchone()
    return dict(row) if row else None


def update_balance(account_id: str, new_balance: float) -> None:
    """Update account balance."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE accounts SET balance = ?, updated_at = ? WHERE account_id = ?",
            (new_balance, now, account_id)
        )


def deduct_balance(account_id: str, amount: float) -> tuple[bool, float, float]:
    """
    Deduct amount from account balance (pre-auth hold).
    Returns (success, old_balance, new_balance).
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT balance FROM accounts WHERE account_id = ?", (account_id,)
        ).fetchone()
        if not row:
            return False, 0.0, 0.0

        old_balance = row["balance"]
        if old_balance < amount:
            return False, old_balance, old_balance

        new_balance = old_balance - amount
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE accounts SET balance = ?, updated_at = ? WHERE account_id = ?",
            (new_balance, now, account_id)
        )
    return True, old_balance, new_balance


def credit_balance(account_id: str, amount: float) -> tuple[float, float]:
    """
    Credit amount to account balance.
    Returns (old_balance, new_balance).
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT balance FROM accounts WHERE account_id = ?", (account_id,)
        ).fetchone()
        old_balance = row["balance"] if row else 0.0
        new_balance = old_balance + amount
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE accounts SET balance = ?, updated_at = ? WHERE account_id = ?",
            (new_balance, now, account_id)
        )
    return old_balance, new_balance


# ══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def create_session(
    user_id: str,
    token_hash: str,
    device_id: str,
    ip_address: str,
    user_agent: Optional[str] = None,
    expires_hours: int = 24
) -> str:
    """Create a new session. Returns session_id."""
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(hours=expires_hours)).isoformat()

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO sessions
               (session_id, user_id, token_hash, device_id, ip_address, user_agent, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, user_id, token_hash, device_id, ip_address, user_agent, now.isoformat(), expires_at)
        )
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Get session by ID."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ? AND is_active = 1",
            (session_id,)
        ).fetchone()
    return dict(row) if row else None


def invalidate_session(session_id: str) -> None:
    """Invalidate a session (logout)."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET is_active = 0 WHERE session_id = ?",
            (session_id,)
        )


def invalidate_all_sessions(user_id: str) -> None:
    """Invalidate all sessions for a user."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET is_active = 0 WHERE user_id = ?",
            (user_id,)
        )


def cleanup_expired_sessions() -> int:
    """Remove expired sessions. Returns count deleted."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE expires_at < ? OR is_active = 0",
            (now,)
        )
    return cursor.rowcount


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def register_device(
    user_id: str,
    device_id: str,
    device_type: str = "WEB",
    device_name: Optional[str] = None
) -> None:
    """Register or update a device for a user."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT device_id FROM device_registry WHERE device_id = ?", (device_id,)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE device_registry SET last_seen = ? WHERE device_id = ?",
                (now, device_id)
            )
        else:
            conn.execute(
                """INSERT INTO device_registry
                   (device_id, user_id, device_name, device_type, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (device_id, user_id, device_name, device_type, now, now)
            )


def is_device_trusted(user_id: str, device_id: str) -> bool:
    """Check if device is trusted for user."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT is_trusted FROM device_registry WHERE user_id = ? AND device_id = ?",
            (user_id, device_id)
        ).fetchone()
    return bool(row and row["is_trusted"])


def is_device_new(user_id: str, device_id: str) -> bool:
    """Check if device is new (not seen before) for user."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT device_id FROM device_registry WHERE user_id = ? AND device_id = ?",
            (user_id, device_id)
        ).fetchone()
    return row is None


def get_user_devices(user_id: str) -> list[dict]:
    """Get all devices for a user."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM device_registry WHERE user_id = ? ORDER BY last_seen DESC",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# REAL-TIME TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_transaction(
    sender_id: str,
    sender_account: str,
    receiver_id: str,
    receiver_account: str,
    amount: float,
    txn_type: TransactionType,
    ip_address: str,
    device_id: str,
    old_balance_sender: float,
    new_balance_sender: float,
    old_balance_receiver: float,
    description: Optional[str] = None,
    reference_id: Optional[str] = None
) -> str:
    """Create a new transaction in INITIATED status. Returns transaction_id."""
    transaction_id = f"TXN{uuid.uuid4().hex[:16].upper()}"
    now = datetime.now(timezone.utc)

    # Calculate step (hour of day + day offset for temporal feature)
    step = now.hour + (now.day * 24)

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO transactions
               (transaction_id, step, type, amount, sender_id, sender_account,
                receiver_id, receiver_account, old_balance_sender, new_balance_sender,
                old_balance_receiver, new_balance_receiver, ip_address, device_id,
                description, reference_id, status, initiated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'INITIATED', ?)""",
            (transaction_id, step, txn_type.value, amount, sender_id, sender_account,
             receiver_id, receiver_account, old_balance_sender, new_balance_sender,
             old_balance_receiver, old_balance_receiver, ip_address, device_id,
             description, reference_id, now.isoformat())
        )
    return transaction_id


def update_transaction_status(transaction_id: str, status: TransactionStatus, **kwargs) -> None:
    """Update transaction status and optional fraud results."""
    kwargs["status"] = status.value
    if status in (TransactionStatus.COMPLETED, TransactionStatus.BLOCKED, TransactionStatus.REVERSED):
        kwargs["completed_at"] = datetime.now(timezone.utc).isoformat()
    if status == TransactionStatus.PENDING_FRAUD:
        kwargs["fraud_check_at"] = datetime.now(timezone.utc).isoformat()

    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [transaction_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE transactions SET {sets} WHERE transaction_id = ?", vals)


def update_transaction_fraud_results(
    transaction_id: str,
    fraud_score: float,
    fraud_label: bool,
    pattern_type: Optional[str],
    pattern_confidence: Optional[float],
    risk_level: str,
    recommended_action: str,
    action_taken: str,
    explanation: Optional[str],
    pipeline_latency_ms: float
) -> None:
    """Update transaction with fraud pipeline results."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE transactions SET
               fraud_score = ?, fraud_label = ?, pattern_type = ?, pattern_confidence = ?,
               risk_level = ?, recommended_action = ?, action_taken = ?, explanation = ?,
               pipeline_latency_ms = ?
               WHERE transaction_id = ?""",
            (fraud_score, 1 if fraud_label else 0, pattern_type, pattern_confidence,
             risk_level, recommended_action, action_taken, explanation,
             pipeline_latency_ms, transaction_id)
        )


def complete_transaction(transaction_id: str, new_balance_receiver: float) -> None:
    """Mark transaction as completed and update receiver balance."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """UPDATE transactions SET
               status = 'COMPLETED', new_balance_receiver = ?, completed_at = ?
               WHERE transaction_id = ?""",
            (new_balance_receiver, now, transaction_id)
        )


def get_transaction(transaction_id: str) -> Optional[dict]:
    """Get transaction by ID."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)
        ).fetchone()
    return dict(row) if row else None


def get_user_transactions(
    user_id: str,
    as_sender: bool = True,
    as_receiver: bool = True,
    limit: int = 50,
    offset: int = 0
) -> list[dict]:
    """Get transactions for a user."""
    conditions = []
    if as_sender:
        conditions.append("sender_id = ?")
    if as_receiver:
        conditions.append("receiver_id = ?")

    where = " OR ".join(conditions) if conditions else "1=0"
    params = [user_id] * len(conditions) + [limit, offset]

    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT * FROM transactions
                WHERE ({where})
                ORDER BY initiated_at DESC LIMIT ? OFFSET ?""",
            params
        ).fetchall()
    return [dict(r) for r in rows]


def get_mule_network_senders(receiver_id: str, lookback_hours: int = 72) -> list[str]:
    """
    Get all unique sender IDs who have sent transactions to receiver_id
    within the lookback window. Used to suspend the entire mule network
    (not just the single sender that triggered detection).
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT sender_id FROM transactions
               WHERE receiver_id = ? AND initiated_at > ?
               ORDER BY initiated_at DESC""",
            (receiver_id, since)
        ).fetchall()
    return [row["sender_id"] for row in rows]


def get_sender_recent_txn_count(sender_id: str, lookback_hours: int = 1) -> int:
    """
    Return the number of transactions the sender has initiated within
    lookback_hours. Used as a DB-backed velocity check to complement
    the in-memory rolling buffer (survives server restarts).
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    with get_conn() as conn:
        row = conn.execute(
            """SELECT COUNT(*) as cnt FROM transactions
               WHERE sender_id = ? AND initiated_at > ?""",
            (sender_id, since)
        ).fetchone()
    return row["cnt"] if row else 0


def get_held_transactions(limit: int = 100) -> list[dict]:
    """Get all HELD transactions for admin review."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT t.*, u1.email as sender_email, u2.email as receiver_email
               FROM transactions t
               JOIN users u1 ON t.sender_id = u1.user_id
               JOIN users u2 ON t.receiver_id = u2.user_id
               WHERE t.status = 'HELD'
               ORDER BY t.initiated_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_transactions(limit: int = 100) -> list[dict]:
    """Get recent transactions for admin monitoring."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT t.*, u1.email as sender_email, u2.email as receiver_email
               FROM transactions t
               JOIN users u1 ON t.sender_id = u1.user_id
               JOIN users u2 ON t.receiver_id = u2.user_id
               ORDER BY t.initiated_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# PAYEE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def add_payee(user_id: str, payee_user_id: str, nickname: Optional[str] = None) -> str:
    """Add a new payee. Returns payee_id."""
    payee_id = f"PAY{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc)

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO payees
               (payee_id, user_id, payee_user_id, nickname, added_at)
               VALUES (?, ?, ?, ?, ?)""",
            (payee_id, user_id, payee_user_id, nickname, now.isoformat())
        )
    return payee_id


def get_payee(user_id: str, payee_user_id: str) -> Optional[dict]:
    """Get payee relationship."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM payees WHERE user_id = ? AND payee_user_id = ?",
            (user_id, payee_user_id)
        ).fetchone()
    return dict(row) if row else None


def get_user_payees(user_id: str) -> list[dict]:
    """Get all payees for a user with receiver details."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT p.*, u.email as payee_email, up.display_name as payee_name, up.business_name
               FROM payees p
               JOIN users u ON p.payee_user_id = u.user_id
               LEFT JOIN user_profiles up ON p.payee_user_id = up.user_id
               WHERE p.user_id = ?
               ORDER BY p.added_at DESC""",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_payee_stats(user_id: str, payee_user_id: str, amount: float) -> None:
    """Update payee transaction stats after successful transaction."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE payees SET
               total_txn_count = total_txn_count + 1,
               total_txn_amount = total_txn_amount + ?,
               is_verified = 1
               WHERE user_id = ? AND payee_user_id = ?""",
            (amount, user_id, payee_user_id)
        )


def remove_payee(user_id: str, payee_id: str) -> bool:
    """Remove a payee. Returns True if deleted."""
    with get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM payees WHERE payee_id = ? AND user_id = ?",
            (payee_id, user_id)
        )
    return cursor.rowcount > 0


# ══════════════════════════════════════════════════════════════════════════════
# VELOCITY TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def get_velocity(user_id: str, window_type: str = "DAILY") -> dict:
    """Get current velocity stats for a user."""
    now = datetime.now(timezone.utc)

    # Calculate window start based on type
    if window_type == "HOURLY":
        window_start = now.replace(minute=0, second=0, microsecond=0)
    elif window_type == "DAILY":
        window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif window_type == "WEEKLY":
        window_start = now - timedelta(days=now.weekday())
        window_start = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
    else:  # MONTHLY
        window_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    with get_conn() as conn:
        row = conn.execute(
            """SELECT txn_count, txn_amount FROM velocity_tracking
               WHERE user_id = ? AND window_type = ? AND window_start = ?""",
            (user_id, window_type, window_start.isoformat())
        ).fetchone()

    if row:
        return {"txn_count": row["txn_count"], "txn_amount": row["txn_amount"]}
    return {"txn_count": 0, "txn_amount": 0.0}


def update_velocity(user_id: str, amount: float) -> None:
    """Update velocity counters after a transaction."""
    now = datetime.now(timezone.utc)

    for window_type in ["HOURLY", "DAILY", "WEEKLY", "MONTHLY"]:
        if window_type == "HOURLY":
            window_start = now.replace(minute=0, second=0, microsecond=0)
        elif window_type == "DAILY":
            window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif window_type == "WEEKLY":
            window_start = now - timedelta(days=now.weekday())
            window_start = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            window_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        with get_conn() as conn:
            existing = conn.execute(
                """SELECT id FROM velocity_tracking
                   WHERE user_id = ? AND window_type = ? AND window_start = ?""",
                (user_id, window_type, window_start.isoformat())
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE velocity_tracking SET
                       txn_count = txn_count + 1, txn_amount = txn_amount + ?, updated_at = ?
                       WHERE id = ?""",
                    (amount, now.isoformat(), existing["id"])
                )
            else:
                conn.execute(
                    """INSERT INTO velocity_tracking
                       (user_id, window_type, window_start, txn_count, txn_amount, updated_at)
                       VALUES (?, ?, ?, 1, ?, ?)""",
                    (user_id, window_type, window_start.isoformat(), amount, now.isoformat())
                )


# ══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def create_compliance_report(
    report_type: str,
    trigger_reason: str,
    transaction_id: Optional[str] = None,
    user_id: Optional[str] = None,
    amount: Optional[float] = None,
    auto_generated: bool = True
) -> str:
    """Create a compliance report (STR/CTR). Returns report_id."""
    report_id = f"RPT{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO compliance_reports
               (report_id, report_type, transaction_id, user_id, amount, trigger_reason, auto_generated, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (report_id, report_type, transaction_id, user_id, amount, trigger_reason, 1 if auto_generated else 0, now)
        )
    return report_id


def get_compliance_reports(report_type: Optional[str] = None, limit: int = 100) -> list[dict]:
    """Get compliance reports, optionally filtered by type."""
    with get_conn() as conn:
        if report_type:
            rows = conn.execute(
                """SELECT * FROM compliance_reports
                   WHERE report_type = ? ORDER BY created_at DESC LIMIT ?""",
                (report_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM compliance_reports ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def should_generate_str(fraud_score: float, action_taken: str, user_id: str) -> tuple[bool, str]:
    """
    Check if Suspicious Transaction Report should be generated.
    Returns (should_generate, reason).
    """
    reasons = []

    # High fraud score
    if fraud_score >= 0.80:
        reasons.append(f"High fraud score: {fraud_score:.2f}")

    # Blocked transaction
    if action_taken == "BLOCK":
        # Check for multiple blocks in 24h
        with get_conn() as conn:
            since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM transactions
                   WHERE sender_id = ? AND action_taken = 'BLOCK' AND initiated_at > ?""",
                (user_id, since)
            ).fetchone()
            if row and row["cnt"] >= 2:
                reasons.append(f"Multiple blocked transactions ({row['cnt']} in 24h)")

    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def should_generate_ctr(txn_type: str, amount: float) -> bool:
    """Check if Cash Transaction Report should be generated (>₹10 lakh cash)."""
    CTR_THRESHOLD = 1000000  # ₹10 lakh
    return txn_type in ("CASH_IN", "CASH_OUT") and amount >= CTR_THRESHOLD


def update_compliance_report_status(
    report_id: str,
    status: str,
    admin_user_id: Optional[str] = None
) -> bool:
    """
    Update compliance report status (PENDING -> SUBMITTED -> ACKNOWLEDGED).
    Records admin who submitted and timestamp.
    Returns True if update succeeded, False if report not found.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        # Check report exists
        row = conn.execute(
            "SELECT status FROM compliance_reports WHERE report_id = ?",
            (report_id,)
        ).fetchone()
        if not row:
            return False

        # Update status
        if status == "SUBMITTED":
            conn.execute(
                """UPDATE compliance_reports
                   SET status = ?, submitted_at = ?, reviewed_by = ?
                   WHERE report_id = ?""",
                (status, now, admin_user_id, report_id)
            )
        else:
            conn.execute(
                """UPDATE compliance_reports
                   SET status = ?, reviewed_by = ?
                   WHERE report_id = ?""",
                (status, admin_user_id, report_id)
            )
    return True


def get_compliance_report(report_id: str) -> Optional[dict]:
    """Get a single compliance report by ID."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM compliance_reports WHERE report_id = ?",
            (report_id,)
        ).fetchone()
    return dict(row) if row else None


def get_compliance_reports_filtered(
    report_type: Optional[str] = None,
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100
) -> list[dict]:
    """Get compliance reports with optional filters."""
    query = "SELECT * FROM compliance_reports WHERE 1=1"
    params: list = []

    if report_type:
        query += " AND report_type = ?"
        params.append(report_type)
    if status:
        query += " AND status = ?"
        params.append(status)
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL OTP FOR PAYMENTS
# ══════════════════════════════════════════════════════════════════════════════

def is_email_otp_enabled(user_id: str) -> bool:
    """Check if user has email OTP enabled for payments."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT email_otp_enabled FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()
    return bool(row and row["email_otp_enabled"])


def set_email_otp_enabled(user_id: str, enabled: bool) -> bool:
    """Enable/disable email OTP for a user. Returns success."""
    with get_conn() as conn:
        result = conn.execute(
            "UPDATE users SET email_otp_enabled = ? WHERE user_id = ?",
            (1 if enabled else 0, user_id)
        )
    return result.rowcount > 0


def create_pending_otp(user_id: str, otp_code: str, transaction_data: dict) -> str:
    """
    Create a pending OTP entry for a transaction.
    OTP is hashed before storage. Returns otp_id.
    """
    from auth.password import hash_password
    
    otp_id = f"OTP{uuid.uuid4().hex[:12].upper()}"
    otp_hash = hash_password(otp_code)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=5)  # OTP expires in 5 minutes
    
    with get_conn() as conn:
        # Clean up expired OTPs for this user
        conn.execute(
            "DELETE FROM pending_otp WHERE user_id = ? OR expires_at < ?",
            (user_id, now.isoformat())
        )
        
        conn.execute(
            """INSERT INTO pending_otp
               (otp_id, user_id, otp_hash, transaction_data, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (otp_id, user_id, otp_hash, json.dumps(transaction_data), now.isoformat(), expires_at.isoformat())
        )
    return otp_id


def verify_and_get_pending_otp(user_id: str, otp_code: str) -> Optional[dict]:
    """
    Verify OTP and return transaction data if valid.
    Returns None if invalid/expired. Deletes OTP after verification.
    """
    from auth.password import verify_password
    
    now = datetime.now(timezone.utc)
    
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM pending_otp 
               WHERE user_id = ? AND expires_at > ?
               ORDER BY created_at DESC LIMIT 1""",
            (user_id, now.isoformat())
        ).fetchone()
        
        if not row:
            return None
        
        otp_data = dict(row)
        
        # Check attempts (max 3)
        if otp_data["attempts"] >= 3:
            conn.execute("DELETE FROM pending_otp WHERE otp_id = ?", (otp_data["otp_id"],))
            return None
        
        # Verify OTP
        if not verify_password(otp_code, otp_data["otp_hash"]):
            conn.execute(
                "UPDATE pending_otp SET attempts = attempts + 1 WHERE otp_id = ?",
                (otp_data["otp_id"],)
            )
            return None
        
        # OTP valid - delete and return transaction data
        conn.execute("DELETE FROM pending_otp WHERE otp_id = ?", (otp_data["otp_id"],))
        
        return json.loads(otp_data["transaction_data"])


def generate_otp() -> str:
    """Generate a 6-digit OTP."""
    return f"{secrets.randbelow(1000000):06d}"


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORT TICKETS
# ══════════════════════════════════════════════════════════════════════════════

def create_support_ticket(user_id: str, subject: str, first_message: str) -> str:
    """Create a support ticket with an initial message. Returns ticket_id."""
    ticket_id = f"TKT{uuid.uuid4().hex[:12].upper()}"
    message_id = f"MSG{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO support_tickets
               (ticket_id, user_id, subject, status, priority, created_at, updated_at)
               VALUES (?, ?, ?, 'OPEN', 'NORMAL', ?, ?)""",
            (ticket_id, user_id, subject, now, now)
        )
        conn.execute(
            """INSERT INTO support_messages
               (message_id, ticket_id, sender_id, sender_role, message, created_at)
               VALUES (?, ?, ?, 'USER', ?, ?)""",
            (message_id, ticket_id, user_id, first_message, now)
        )
    return ticket_id


def get_support_ticket(ticket_id: str) -> Optional[dict]:
    """Get a single support ticket by ID."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT t.*, u.email as user_email, p.display_name
               FROM support_tickets t
               JOIN users u ON t.user_id = u.user_id
               LEFT JOIN user_profiles p ON t.user_id = p.user_id
               WHERE t.ticket_id = ?""",
            (ticket_id,)
        ).fetchone()
    return dict(row) if row else None


def get_user_support_tickets(user_id: str) -> list[dict]:
    """Get all support tickets for a user."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT t.*, u.email as user_email, p.display_name
               FROM support_tickets t
               JOIN users u ON t.user_id = u.user_id
               LEFT JOIN user_profiles p ON t.user_id = p.user_id
               WHERE t.user_id = ?
               ORDER BY t.updated_at DESC""",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_support_tickets(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    """Get all support tickets (admin view)."""
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                """SELECT t.*, u.email as user_email, p.display_name,
                          u.account_status as user_account_status
                   FROM support_tickets t
                   JOIN users u ON t.user_id = u.user_id
                   LEFT JOIN user_profiles p ON t.user_id = p.user_id
                   WHERE t.status = ?
                   ORDER BY t.updated_at DESC LIMIT ?""",
                (status, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT t.*, u.email as user_email, p.display_name,
                          u.account_status as user_account_status
                   FROM support_tickets t
                   JOIN users u ON t.user_id = u.user_id
                   LEFT JOIN user_profiles p ON t.user_id = p.user_id
                   ORDER BY t.updated_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_ticket_messages(ticket_id: str) -> list[dict]:
    """Get all messages for a ticket, ordered oldest first."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT m.*, p.display_name as sender_name, u.email as sender_email
               FROM support_messages m
               JOIN users u ON m.sender_id = u.user_id
               LEFT JOIN user_profiles p ON m.sender_id = p.user_id
               WHERE m.ticket_id = ?
               ORDER BY m.created_at ASC""",
            (ticket_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def add_ticket_message(ticket_id: str, sender_id: str, sender_role: str, message: str) -> str:
    """Add a message to a ticket. Returns message_id."""
    message_id = f"MSG{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO support_messages
               (message_id, ticket_id, sender_id, sender_role, message, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (message_id, ticket_id, sender_id, sender_role, message, now)
        )
        # Update ticket's updated_at and auto-set IN_PROGRESS if admin replied
        if sender_role == "ADMIN":
            conn.execute(
                "UPDATE support_tickets SET updated_at = ?, status = 'IN_PROGRESS' WHERE ticket_id = ? AND status = 'OPEN'",
                (now, ticket_id)
            )
        else:
            conn.execute(
                "UPDATE support_tickets SET updated_at = ? WHERE ticket_id = ?",
                (now, ticket_id)
            )
    return message_id


def update_ticket_status(ticket_id: str, status: str) -> None:
    """Update ticket status."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        update = "UPDATE support_tickets SET status = ?, updated_at = ?"
        params = [status, now]
        if status == "RESOLVED":
            update += ", resolved_at = ?"
            params.append(now)
        params.append(ticket_id)
        conn.execute(f"{update} WHERE ticket_id = ?", params)
