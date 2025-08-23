"""
Drift Detection MCP Server
- Works when spawned by an MCP client over STDIO (ADK/LLM agent).
- Also works as a long‑lived standalone process (TCP fallback) for containerized deployments.

Key features
- Tools:
  * detect_drift: compute simple drift metrics between a reference and current sample
  * set_baseline: store/update a named reference baseline (SQLite or in‑memory)
  * get_baseline: fetch a stored baseline by name
  * list_baselines: list stored baselines
  * delete_baseline: remove a stored baseline
  * health: quick liveness probe
- Metrics implemented in pure Python (no heavy deps):
  * KS statistic (empirical CDF distance)
  * Mean & variance shift
  * PSI (Population Stability Index) with quantile binning
- Lazy DB init; if SQLite path not writable/available it gracefully falls back to in‑memory storage.

Environment variables (optional)
- DRIFT_DB: path to SQLite file (default: ./data/drift.db)
- DRIFT_LOG_LEVEL: logging level (DEBUG/INFO/WARNING/ERROR) default INFO
- MCP_TCP_HOST: host for TCP fallback (default 0.0.0.0)
- MCP_TCP_PORT: port for TCP fallback (default 6101)

"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# MCP server SDK
from mcp.server import stdio
from mcp.server.fastmcp import FastMCP

# ----------------------------
# Logging setup
# ----------------------------
LOG_LEVEL = os.getenv("DRIFT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("drift_mcp")

# ----------------------------
# Persistence (SQLite or in-memory)
# ----------------------------
DEFAULT_DB_PATH = os.getenv("DRIFT_DB", os.path.join(os.getcwd(), "data", "drift.db"))

@dataclass
class Store:
    conn: Optional[sqlite3.Connection]
    mem: Dict[str, List[float]]


def ensure_db_dir(path: str) -> None:
    try:
        dir_ = os.path.dirname(path)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create DB directory %s: %s (falling back to memory)", path, e)


def try_open_sqlite(path: str) -> Optional[sqlite3.Connection]:
    try:
        ensure_db_dir(path)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baselines (
                name TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return conn
    except Exception as e:
        logger.warning("SQLite disabled (%s). Using in-memory store only.", e)
        return None


STORE = Store(conn=try_open_sqlite(DEFAULT_DB_PATH), mem={})


def db_get(name: str) -> Optional[List[float]]:
    if STORE.conn is None:
        return STORE.mem.get(name)
    cur = STORE.conn.execute("SELECT payload FROM baselines WHERE name = ?", (name,))
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def db_set(name: str, values: List[float]) -> None:
    if STORE.conn is None:
        STORE.mem[name] = list(values)
        return
    payload = json.dumps(list(values))
    STORE.conn.execute(
        """
        INSERT INTO baselines(name, payload)
        VALUES(?, ?)
        ON CONFLICT(name) DO UPDATE SET
            payload=excluded.payload,
            updated_at=CURRENT_TIMESTAMP
        """,
        (name, payload),
    )
    STORE.conn.commit()


def db_delete(name: str) -> bool:
    if STORE.conn is None:
        return STORE.mem.pop(name, None) is not None
    cur = STORE.conn.execute("DELETE FROM baselines WHERE name = ?", (name,))
    STORE.conn.commit()
    return cur.rowcount > 0


def db_list() -> List[str]:
    if STORE.conn is None:
        return sorted(STORE.mem.keys())
    cur = STORE.conn.execute("SELECT name FROM baselines ORDER BY name")
    return [r[0] for r in cur.fetchall()]


# ----------------------------
# Drift metrics (pure Python)
# ----------------------------

Number = float  # simplify type hints


def _sorted(values: List[Number]) -> List[Number]:
    return sorted(v for v in values if v is not None)


def ks_statistic(ref: List[Number], cur: List[Number]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (ECDF max distance), O(n log n).
    Pure Python implementation to avoid heavy deps. Assumes numeric lists.
    """
    r = _sorted(ref)
    c = _sorted(cur)
    if not r or not c:
        return 0.0
    i = j = 0
    nr = len(r)
    nc = len(c)
    d_max = 0.0
    while i < nr and j < nc:
        if r[i] <= c[j]:
            i += 1
        else:
            j += 1
        # ECDF values are i/nr and j/nc at current step
        d = abs(i / nr - j / nc)
        if d > d_max:
            d_max = d
    # also check tail ends
    d_max = max(d_max, abs(1 - j / nc), abs(i / nr - 1))
    return d_max


def mean_var_shift(ref: List[Number], cur: List[Number]) -> Dict[str, Number]:
    if not ref or not cur:
        return {"mean_diff": 0.0, "var_ratio": 1.0}
    m_r = statistics.fmean(ref)
    m_c = statistics.fmean(cur)
    var_r = statistics.pvariance(ref) if len(ref) > 1 else 0.0
    var_c = statistics.pvariance(cur) if len(cur) > 1 else 0.0
    var_ratio = (var_c / var_r) if var_r > 0 else float("inf") if var_c > 0 else 1.0
    return {"mean_diff": m_c - m_r, "var_ratio": var_ratio}


def psi(ref: List[Number], cur: List[Number], bins: int = 10) -> float:
    """Population Stability Index using quantile bins on reference.
    PSI = sum( (p_i - q_i) * ln(p_i / q_i) ), with epsilon smoothing.
    """
    r = _sorted(ref)
    c = _sorted(cur)
    if not r or not c:
        return 0.0

    # Build quantile cutpoints from reference
    def qtile(data: List[Number], q: float) -> Number:
        idx = max(0, min(len(data) - 1, int(round(q * (len(data) - 1)))))
        return data[idx]

    cuts = [qtile(r, i / bins) for i in range(1, bins)]  # len = bins-1
    # Bin function
    def count_in_bin(x: Number, cuts_: List[Number]) -> int:
        # returns bin index 0..bins-1
        lo, hi = 0, len(cuts_)
        while lo < hi:
            mid = (lo + hi) // 2
            if x <= cuts_[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    r_bins = [0] * bins
    c_bins = [0] * bins
    for x in r:
        r_bins[count_in_bin(x, cuts)] += 1
    for x in c:
        c_bins[count_in_bin(x, cuts)] += 1

    eps = 1e-6
    psi_val = 0.0
    for rb, cb in zip(r_bins, c_bins):
        p = rb / len(r)
        q = cb / len(c)
        p = max(p, eps)
        q = max(q, eps)
        psi_val += (p - q) * (math_log(p / q))
    return psi_val


def math_log(x: float) -> float:
    import math
    return math.log(x)


def decide_drift(ks: float, psi_val: float, mean_diff: float, var_ratio: float,
                 ks_thresh: float = 0.2, psi_thresh: float = 0.1,
                 mean_thresh: float = 0.25, var_ratio_thresh: float = 1.5) -> Dict[str, Any]:
    """Simple rules to flag drift. Tune thresholds for your domain."""
    flags = {
        "ks_exceeds": ks >= ks_thresh,
        "psi_exceeds": psi_val >= psi_thresh,
        "mean_exceeds": abs(mean_diff) >= mean_thresh,
        "var_exceeds": (var_ratio >= var_ratio_thresh) or (var_ratio <= (1 / var_ratio_thresh)),
    }
    drifted = any(flags.values())
    return {"drift": drifted, "rules": flags}


# ----------------------------
# MCP App & Tools
# ----------------------------
app = FastMCP("drift_detection_mcp_server", version="0.1.0")


@app.tool("health", desc="Liveness/health check for the Drift MCP server.")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "storage": "sqlite" if STORE.conn else "memory",
        "baselines": len(db_list()),
    }


@app.tool(
    "set_baseline",
    desc=(
        "Store or update a named reference baseline of numeric values. "
        "Params: name (str), values (List[float])."
    ),
)
def set_baseline(name: str, values: List[float]) -> Dict[str, Any]:
    if not isinstance(values, list) or not all(isinstance(v, (int, float)) for v in values):
        raise ValueError("values must be a list of numbers")
    db_set(name, [float(v) for v in values])
    return {"ok": True, "name": name, "count": len(values)}


@app.tool(
    "get_baseline",
    desc="Fetch a stored baseline by name. Returns {name, values} or raises if missing.",
)
def get_baseline(name: str) -> Dict[str, Any]:
    values = db_get(name)
    if values is None:
        raise ValueError(f"baseline not found: {name}")
    return {"name": name, "values": values, "count": len(values)}


@app.tool("list_baselines", desc="List the names of all stored baselines.")
def list_baselines() -> Dict[str, Any]:
    return {"baselines": db_list()}


@app.tool("delete_baseline", desc="Delete a baseline by name. Returns {deleted: bool}.")
def delete_baseline(name: str) -> Dict[str, Any]:
    deleted = db_delete(name)
    return {"deleted": bool(deleted), "name": name}


@app.tool(
    "detect_drift",
    desc=(
        "Compute drift between a reference and current sample. "
        "You can pass either baseline_name (str) or explicit reference list. "
        "Params: baseline_name (optional str), reference (optional List[float]), current (List[float]), "
        "ks_thresh, psi_thresh, mean_thresh, var_ratio_thresh (optional floats)."
    ),
)
def detect_drift(
    current: List[float],
    baseline_name: Optional[str] = None,
    reference: Optional[List[float]] = None,
    ks_thresh: float = 0.2,
    psi_thresh: float = 0.1,
    mean_thresh: float = 0.25,
    var_ratio_thresh: float = 1.5,
    psi_bins: int = 10,
) -> Dict[str, Any]:
    # Resolve reference
    ref_vals: Optional[List[float]] = None
    if baseline_name:
        ref_vals = db_get(baseline_name)
        if ref_vals is None:
            raise ValueError(f"baseline not found: {baseline_name}")
    elif reference is not None:
        if not isinstance(reference, list) or not all(isinstance(v, (int, float)) for v in reference):
            raise ValueError("reference must be a list of numbers")
        ref_vals = [float(v) for v in reference]
    else:
        raise ValueError("Provide either baseline_name or reference array")

    if not isinstance(current, list) or not all(isinstance(v, (int, float)) for v in current):
        raise ValueError("current must be a list of numbers")

    cur_vals = [float(v) for v in current]

    ks = ks_statistic(ref_vals, cur_vals)
    mv = mean_var_shift(ref_vals, cur_vals)
    psi_val = psi(ref_vals, cur_vals, bins=int(max(3, psi_bins)))
    decision = decide_drift(ks, psi_val, mv["mean_diff"], mv["var_ratio"], ks_thresh, psi_thresh, mean_thresh, var_ratio_thresh)

    return {
        "metrics": {
            "ks": ks,
            "psi": psi_val,
            "mean_diff": mv["mean_diff"],
            "var_ratio": mv["var_ratio"],
        },
        "decision": decision,
        "sizes": {"reference": len(ref_vals or []), "current": len(cur_vals)},
        "used_baseline": baseline_name if baseline_name else None,
    }


# ----------------------------
# Entrypoint: stdio when spawned, TCP fallback when standalone
# ----------------------------
async def run_stdio() -> None:
    logger.info("Drift Detection MCP Stdio Server: Starting handshake with client...")
    async with stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream)
    logger.info("Drift Detection MCP Stdio Server: Run loop finished or client disconnected.")


async def run_tcp(host: str, port: int) -> None:
    """Optional TCP fallback: keeps the process alive so supervisors can manage it.
    We don't expose a TCP MCP transport here to keep dependencies minimal; this just
    prevents crashes when no stdio is available (e.g., started from start.sh).
    """
    import anyio
    logger.warning("No STDIO available; running in standalone keep-alive mode on %s:%s (no network server)", host, port)
    # If you later want real TCP transport, replace with mcp.server.tcp.serve(...)
    await anyio.sleep_forever()


async def main() -> None:
    # Lazy vacuum to ensure DB reachable (no-op if memory store)
    try:
        if STORE.conn is not None:
            STORE.conn.execute("VACUUM")
    except Exception as e:
        logger.debug("SQLite VACUUM skipped: %s", e)

    # Decide whether stdio is available
    use_stdio = False
    try:
        import sys
        use_stdio = bool(sys.stdin) and (not sys.stdin.closed)
    except Exception:
        use_stdio = False

    if use_stdio:
        await run_stdio()
    else:
        host = os.getenv("MCP_TCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_TCP_PORT", "6101"))
        await run_tcp(host, port)


if __name__ == "__main__":
    logger.info("Creating Drift Detection MCP Server instance...")
    logger.info("Launching Drift Detection MCP Server via stdio or standalone fallback...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Drift Detection MCP Server interrupted; shutting down.")
    except Exception as e:
        logger.exception("Fatal error in Drift Detection MCP Server: %s", e)
