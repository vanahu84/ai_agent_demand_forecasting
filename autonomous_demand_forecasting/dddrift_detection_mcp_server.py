"""
Drift Detection MCP Server
- Works when spawned by an MCP client over STDIO (ADK/LLM agent).
- Also works as a long‑lived standalone process (TCP fallback) for containerized deployments.
"""
from __future__ import annotations
import asyncio, json, logging, os, sqlite3, statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from mcp.server import stdio
from mcp.server.fastmcp import FastMCP

LOG_LEVEL = os.getenv("DRIFT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("drift_mcp")
DEFAULT_DB_PATH = os.getenv("DRIFT_DB", os.path.join(os.getcwd(), "data", "drift.db"))

@dataclass
class Store:
    conn: Optional[sqlite3.Connection]
    mem: Dict[str, List[float]]

def ensure_db_dir(path: str):
    try:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create DB directory %s: %s", path, e)

def try_open_sqlite(path: str):
    try:
        ensure_db_dir(path)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS baselines (name TEXT PRIMARY KEY, payload TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        conn.commit()
        return conn
    except Exception as e:
        logger.warning("SQLite disabled: %s", e)
        return None

STORE = Store(conn=try_open_sqlite(DEFAULT_DB_PATH), mem={})

def db_get(name):
    if STORE.conn is None:
        return STORE.mem.get(name)
    cur = STORE.conn.execute("SELECT payload FROM baselines WHERE name=?", (name,))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None

def db_set(name, values):
    if STORE.conn is None:
        STORE.mem[name] = list(values); return
    payload = json.dumps(list(values))
    STORE.conn.execute("INSERT INTO baselines(name,payload) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET payload=excluded.payload, updated_at=CURRENT_TIMESTAMP", (name,payload))
    STORE.conn.commit()

def db_delete(name):
    if STORE.conn is None:
        return STORE.mem.pop(name, None) is not None
    cur = STORE.conn.execute("DELETE FROM baselines WHERE name=?", (name,)); STORE.conn.commit(); return cur.rowcount>0

def db_list():
    if STORE.conn is None:
        return sorted(STORE.mem.keys())
    return [r[0] for r in STORE.conn.execute("SELECT name FROM baselines ORDER BY name").fetchall()]

def _sorted(values): return sorted(v for v in values if v is not None)

def ks_statistic(ref, cur):
    r,c=_sorted(ref),_sorted(cur)
    if not r or not c: return 0.0
    i=j=0; nr, nc=len(r),len(c); d_max=0.0
    while i<nr and j<nc:
        if r[i]<=c[j]: i+=1
        else: j+=1
        d=abs(i/nr-j/nc)
        if d>d_max: d_max=d
    d_max=max(d_max,abs(1-j/nc),abs(i/nr-1))
    return d_max

def mean_var_shift(ref,cur):
    if not ref or not cur: return {"mean_diff":0.0,"var_ratio":1.0}
    m_r=statistics.fmean(ref); m_c=statistics.fmean(cur)
    var_r=statistics.pvariance(ref) if len(ref)>1 else 0.0
    var_c=statistics.pvariance(cur) if len(cur)>1 else 0.0
    var_ratio=(var_c/var_r) if var_r>0 else float("inf") if var_c>0 else 1.0
    return {"mean_diff":m_c-m_r,"var_ratio":var_ratio}

def psi(ref,cur,bins=10):
    r,c=_sorted(ref),_sorted(cur)
    if not r or not c: return 0.0
    def qtile(data,q):
        idx=max(0,min(len(data)-1,int(round(q*(len(data)-1)))))
        return data[idx]
    cuts=[qtile(r,i/bins) for i in range(1,bins)]
    def count_in_bin(x,cuts_):
        lo,hi=0,len(cuts_)
        while lo<hi:
            mid=(lo+hi)//2
            if x<=cuts_[mid]: hi=mid
            else: lo=mid+1
        return lo
    r_bins=[0]*bins; c_bins=[0]*bins
    for x in r: r_bins[count_in_bin(x,cuts)]+=1
    for x in c: c_bins[count_in_bin(x,cuts)]+=1
    eps=1e-6; psi_val=0.0
    import math
    for rb,cb in zip(r_bins,c_bins):
        p=max(rb/len(r),eps); q=max(cb/len(c),eps)
        psi_val+=(p-q)*math.log(p/q)
    return psi_val

def decide_drift(ks,psi_val,mean_diff,var_ratio,ks_thresh=0.2,psi_thresh=0.1,mean_thresh=0.25,var_ratio_thresh=1.5):
    flags={"ks_exceeds":ks>=ks_thresh,"psi_exceeds":psi_val>=psi_thresh,"mean_exceeds":abs(mean_diff)>=mean_thresh,"var_exceeds":(var_ratio>=var_ratio_thresh) or (var_ratio<=1/var_ratio_thresh)}
    return {"drift":any(flags.values()),"rules":flags}

app=FastMCP("drift_detection_mcp_server",version="0.1.0")

@app.tool("health")
def health()->Dict[str,Any]:
    """Liveness/health check for the Drift MCP server."""
    return {"status":"ok","storage":"sqlite" if STORE.conn else "memory","baselines":len(db_list())}

@app.tool("set_baseline")
def set_baseline(name:str,values:List[float])->Dict[str,Any]:
    """Store or update a named reference baseline of numeric values."""
    if not isinstance(values,list) or not all(isinstance(v,(int,float)) for v in values):
        raise ValueError("values must be list of numbers")
    db_set(name,[float(v) for v in values]); return {"ok":True,"name":name,"count":len(values)}

@app.tool("get_baseline")
def get_baseline(name:str)->Dict[str,Any]:
    """Fetch a stored baseline by name."""
    vals=db_get(name)
    if vals is None: raise ValueError(f"baseline not found: {name}")
    return {"name":name,"values":vals,"count":len(vals)}

@app.tool("list_baselines")
def list_baselines()->Dict[str,Any]:
    """List the names of all stored baselines."""
    return {"baselines":db_list()}

@app.tool("delete_baseline")
def delete_baseline(name:str)->Dict[str,Any]:
    """Delete a baseline by name."""
    return {"deleted":bool(db_delete(name)),"name":name}

@app.tool("detect_drift")
def detect_drift(current:List[float],baseline_name:Optional[str]=None,reference:Optional[List[float]]=None,ks_thresh:float=0.2,psi_thresh:float=0.1,mean_thresh:float=0.25,var_ratio_thresh:float=1.5,psi_bins:int=10)->Dict[str,Any]:
    """Compute drift metrics between a reference and current sample."""
    if baseline_name:
        ref_vals=db_get(baseline_name)
        if ref_vals is None: raise ValueError(f"baseline not found: {baseline_name}")
    elif reference is not None:
        if not isinstance(reference,list) or not all(isinstance(v,(int,float)) for v in reference):
            raise ValueError("reference must be a list of numbers")
        ref_vals=[float(v) for v in reference]
    else:
        raise ValueError("Provide either baseline_name or reference")
    if not isinstance(current,list) or not all(isinstance(v,(int,float)) for v in current):
        raise ValueError("current must be list of numbers")
    cur_vals=[float(v) for v in current]
    ks=ks_statistic(ref_vals,cur_vals); mv=mean_var_shift(ref_vals,cur_vals); psi_val=psi(ref_vals,cur_vals,bins=max(3,int(psi_bins)))
    decision=decide_drift(ks,psi_val,mv["mean_diff"],mv["var_ratio"],ks_thresh,psi_thresh,mean_thresh,var_ratio_thresh)
    return {"metrics":{"ks":ks,"psi":psi_val,"mean_diff":mv["mean_diff"],"var_ratio":mv["var_ratio"]},"decision":decision,"sizes":{"reference":len(ref_vals),"current":len(cur_vals)},"used_baseline":baseline_name}

async def run_stdio():
    logger.info("Drift Detection MCP Stdio Server: Starting handshake with client...")
    async with stdio.stdio_server() as (r,w):
        await app.run(r,w)
    logger.info("Drift Detection MCP Stdio Server: Run loop finished or client disconnected.")

async def run_tcp(host:str,port:int):
    import anyio
    logger.warning("No STDIO; running standalone keep-alive on %s:%s",host,port)
    await anyio.sleep_forever()

async def main():
    try:
        if STORE.conn is not None:
            STORE.conn.execute("VACUUM")
    except Exception as e:
        logger.debug("SQLite VACUUM skipped: %s",e)
    use_stdio=False
    try:
        import sys
        use_stdio=bool(sys.stdin) and (not sys.stdin.closed)
    except Exception: use_stdio=False
    if use_stdio: await run_stdio()
    else:
        host=os.getenv("MCP_TCP_HOST","0.0.0.0"); port=int(os.getenv("MCP_TCP_PORT","6101")); await run_tcp(host,port)

if __name__=="__main__":
    logger.info("Creating Drift Detection MCP Server instance...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down.")
    except Exception as e:
        logger.exception("Fatal error: %s",e)
