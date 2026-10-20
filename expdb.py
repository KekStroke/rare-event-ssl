"""expdb: an experiment result is only a result together with the contract it was produced under.

One SQLite table `runs` (kind = pretrain | probe | prep | analysis). Every row carries
  fingerprint   hardware / driver / torch / python  (fp_id = comparability key for timings)
  code_hash     sha256 of common.py+prep.py+pretrain.py+probe.py as they were when the run was recorded
  data_fp       md5 of the data artefacts the run consumed (lab16k, mel_stats, targets_*)
  args / protocol  everything needed to re-run
  curve / telemetry / results  the full payload, never summarised away
  valid / invalid_reason   invariants: a row that broke its contract is kept, flagged, never compared
  hypothesis    H<n> from knowledge/hypotheses.json
Usage from scripts: record(kind, name, **fields). CLI: python expdb.py backfill | table | findings
"""
import hashlib, json, os, platform, sqlite3, subprocess, sys, time, glob
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / 'tsssl'
DB = ROOT / 'expdb' / 'expdb.sqlite'
CODE = ['common.py', 'prep.py', 'pretrain.py', 'probe.py']
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
  fp_id TEXT, fp_json TEXT, code_hash TEXT, code_snapshot TEXT, data_fp TEXT,
  variant TEXT, block TEXT, stack INTEGER, seed INTEGER, steps INTEGER, args TEXT,
  params_m REAL, final_loss REAL, final_acc REAL, minutes REAL, steps_per_s REAL, curve TEXT,
  feats TEXT, protocol TEXT, f1_cls_0p05 REAL, f1_cls_0p2 REAL, f1_cls_1p0 REAL, f1_any_0p2 REAL, f1_any_1p0 REAL,
  onset_p50 REAL, onset_p90 REAL, recall_op_recall REAL, recall_op_precision REAL, results TEXT,
  telemetry TEXT, gpu_shared INTEGER, ckpt TEXT, ckpt_md5 TEXT,
  valid INTEGER NOT NULL DEFAULT 1, invalid_reason TEXT, hypothesis TEXT, notes TEXT
);
CREATE INDEX IF NOT EXISTS runs_key ON runs(kind, name, fp_id);
"""

def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ''

def fingerprint():
    q = sh("nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm,clocks.max.memory,power.limit --format=csv,noheader")
    g = [x.strip() for x in q.split(',')] if q else [None] * 6
    fp = dict(gpu=g[0], driver=g[1], cc=g[2], max_sm_mhz=g[3], max_mem_mhz=g[4], power_limit_w=g[5],
              cpu_threads=os.cpu_count(), os=platform.platform(), python=platform.python_version(), wsl=True)
    try:
        import torch; fp.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=str(torch.backends.cudnn.version()))
    except Exception: pass
    fid = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:12]
    return fid, fp

def telemetry():
    q = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader")
    k = ['util_pct', 'mem_mib', 'power_w', 'sm_mhz', 'mem_mhz', 'temp_c', 'throttle']
    return dict(zip(k, [x.strip() for x in q.split(',')])) if q else {}

def code_hash():
    h = hashlib.sha256(); snap = {}
    for f in CODE:
        p = ROOT / f
        if p.exists(): t = p.read_text(); h.update(t.encode()); snap[f] = t
    return h.hexdigest()[:12], snap

def md5(p, n=1 << 24):
    p = Path(p)
    if not p.exists(): return None
    h = hashlib.md5()
    with open(p, 'rb') as f: h.update(f.read(n))          # first 16 MB + size: fast, stable
    h.update(str(p.stat().st_size).encode()); return h.hexdigest()[:12]

def data_fp(names):
    return json.dumps({n: md5(ROOT / 'data' / n) for n in names}, sort_keys=True)

def gpu_jobs():
    """Other python processes on the GPU right now (contract: shared GPU => minutes not comparable)."""
    q = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    return [l for l in q.splitlines() if l.strip()]

def connect():
    DB.parent.mkdir(exist_ok=True); c = sqlite3.connect(DB); c.executescript(SCHEMA); return c

def record(kind, name, **f):
    fid, fp = fingerprint(); ch, snap = code_hash()
    row = dict(ts=datetime.now(timezone.utc).isoformat(timespec='seconds'), kind=kind, name=name, fp_id=fid,
               fp_json=json.dumps(fp), code_hash=ch, code_snapshot=json.dumps(snap), telemetry=json.dumps(telemetry()),
               gpu_shared=int(len(gpu_jobs()) > 1))
    for k, v in f.items():
        row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    row.setdefault('valid', 1)
    with connect() as c:
        cols = ','.join(row); q = ','.join('?' * len(row))
        cur = c.execute(f'INSERT INTO runs ({cols}) VALUES ({q})', list(row.values()))
        return cur.lastrowid

# ----------------------------------------------------------------------------- invariants
def pretrain_invariants(curve, steps_target, final_loss):
    if not curve: return 'no training log'
    if curve[-1]['step'] < steps_target: return f'stopped at {curve[-1]["step"]}/{steps_target}'
    if final_loss is None or final_loss != final_loss: return 'non-finite loss'
    return None

def probe_invariants(res):
    pf = res.get('per_fold', [])
    if len(pf) != 3: return f'{len(pf)} folds'
    if any(d['n_pred'] == 0 for d in pf): return 'a fold predicted nothing'
    return None

# ----------------------------------------------------------------------------- hooks used by scripts
def record_pretrain(args, curve, params_m, ckpt, minutes, extra_notes=None):
    steps = args['steps']; fl = curve[-1]['loss'] if curve else None
    inv = pretrain_invariants(curve, steps, fl)
    data = ['lab16k.npz', 'mel_stats.npz'] + ([f"targets_{args['variant']}.npy"] if args['variant'] not in ('d2v', 'rq') else []) \
        + (['targets_kmmel.npy', 'targets_kmaves.npy'] if args['variant'] == 'kmboth' else [])
    return record('pretrain', args['variant'] + args.get('tag', ''), variant=args['variant'], block=args.get('block', 'vanilla'),
                  stack=int(os.environ.get('TSSSL_STACK', 2)), seed=args.get('seed', 0), steps=steps, args=args,
                  params_m=params_m, final_loss=fl, final_acc=curve[-1]['acc'] if curve else None, minutes=minutes,
                  steps_per_s=round(steps / (minutes * 60), 2) if minutes else None, curve=curve, data_fp=data_fp(data),
                  ckpt=str(ckpt), ckpt_md5=md5(ckpt), valid=int(inv is None), invalid_reason=inv, notes=extra_notes)

def record_probe(res, protocol, feats):
    inv = probe_invariants(res); p = res['pooled']; t = res.get('timing', {}); r = res.get('recall_op', {})
    ck = feats if feats.endswith('.pt') else None
    return record('probe', res['name'], feats=feats, protocol=protocol, results=res,
                  stack=int(os.environ.get('TSSSL_STACK', 2)),
                  f1_cls_0p05=p['cls@0.05'][2], f1_cls_0p2=p['cls@0.2'][2], f1_cls_1p0=p['cls@1.0'][2],
                  f1_any_0p2=p['any@0.2'][2], f1_any_1p0=p['any@1.0'][2],
                  onset_p50=(t.get('abs_onset_p50_75_90') or [None])[0], onset_p90=(t.get('abs_onset_p50_75_90') or [None] * 3)[2],
                  recall_op_recall=r.get('recall'), recall_op_precision=r.get('precision'),
                  ckpt=ck, ckpt_md5=md5(ck) if ck else None, data_fp=data_fp(['mel_stats.npz']),
                  valid=int(inv is None), invalid_reason=inv)

PROTOCOL = dict(task='Avacha killer-whale call events, 17 recordings, 613 events, 11 types', sr=16000,
                folds='3 recording-held-out folds 6/6/5 (seed-0 shuffle, lists in common.FOLDS)',
                dev='first recording of each of the two training folds -> threshold grid search',
                probe='frozen features, softmax-weighted layer sum + LayerNorm + linear, 3000 AdamW steps, bs 8192 tokens, sqrt-inverse class weights',
                decoder='p_call = 1 - p_noise, median 5 tokens, two thresholds detect/edge from grid {.5..9}x{.2..5} maximising class-aware F1@0.2 on dev, min_dur 0.15 s',
                scorer='greedy onset-nearest matching, onset collar tau, offset tolerance max(tau, 0.5*dur), pooled over the 3 test folds',
                recall_op='thresholds maximising class-agnostic recall s.t. precision>=0.5 at tau=0.5 on dev; FAR/h over 1.22 h',
                caveat='single seed per model; numbers not comparable to the ICLR paper protocol (1 s windows / 0.1 s hop, nested CV)')

# ----------------------------------------------------------------------------- backfill from logs
def backfill():
    """Reconstruct rows for runs made before expdb existed. Timings carry gpu_shared=1 (manual probes ran alongside)."""
    import torch
    n = 0
    for log in sorted(glob.glob(str(ROOT / 'logs' / 'pretrain_*.log'))):
        name = Path(log).stem[len('pretrain_'):]
        curve = [json.loads(l) for l in open(log) if l.startswith('{"step"')]
        import re
        ck = sorted([p for p in glob.glob(str(ROOT / 'ckpt' / f'{name}_*.pt')) if re.fullmatch(rf'{re.escape(name)}_\d+\.pt', Path(p).name)],
                    key=lambda p: int(p.rsplit('_', 1)[1][:-3]))
        if not ck: continue
        a = torch.load(ck[-1], map_location='cpu')['args']; a.setdefault('block', 'vanilla'); a.setdefault('seed', 0); a.setdefault('tag', '')
        stack = 1 if name.endswith('_s1') or '_s1_' in name else 4 if name.endswith('_s4') else 2
        os.environ['TSSSL_STACK'] = str(stack)
        params = 17.81 if a['block'] == 'modern' else 14.84
        running = bool(curve) and curve[-1]['step'] < a['steps']
        record_pretrain(a, curve, params, ck[-1], curve[-1]['min'] if curve else None,
                        extra_notes=('IN PROGRESS at backfill time; ' if running else '') + 'backfilled from log; code_hash is the code at backfill time; GPU shared with manual probes during the run')
        n += 1
    for rj in sorted(glob.glob(str(ROOT / 'results' / '*.json'))):
        res = json.load(open(rj))
        if 'name' not in res: continue
        stack = 1 if res['name'].endswith('_s1') or '_s1_' in res['name'] else 4 if res['name'].endswith('_s4') else 2
        os.environ['TSSSL_STACK'] = str(stack)
        proto = dict(PROTOCOL); proto['probe_steps'] = 8000 if res['name'] == 'aves_8k' else 3000
        record_probe(res, proto, res['feats']); n += 1
    print('backfilled rows', n)

def table():
    with connect() as c:
        for r in c.execute("SELECT id, ts, kind, name, valid, f1_cls_0p2, f1_any_0p2, onset_p50, final_loss, final_acc, minutes, gpu_shared, code_hash FROM runs ORDER BY id"):
            print(r)

if __name__ == '__main__':
    {'backfill': backfill, 'table': table}[sys.argv[1]]()
