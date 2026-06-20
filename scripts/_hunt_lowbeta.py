"""Hunt: DEFENSIVE LOW-BETA (BAB L/S + long-biased sleeve). Full audit gauntlet.

backtest_low_beta requires every symbol on every common day -> can't include delisted
names (exit mid-sample) NOR even heterogeneous-start survivors. So:
  - clean survivor universe = midcap_sip symbols with full 1255-day grid (227 syms, 5yr).
  - for the SURVIVORSHIP PIT (must handle entry/exit), use a PIT-aware re-implementation that
    mirrors backtest_low_beta's exact logic (beta to SPY, long-low/short-high, dollar-neutral,
    same lookback/rebalance/cost) but trades each name only on days it has data.
"""
import json, time
from pathlib import Path
from collections import Counter
import numpy as np
from alpca.backtest.low_beta import backtest_low_beta
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of
from alpca.backtest.leg_gate import evaluate_leg_candidate

DATA = Path("/Volumes/My Passport/AlpcaData")
def load(d):
    return {p.name.split('_1day_')[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in (DATA/d).glob('*_1day_bars.jsonl')}
def year_of(ts): return time.gmtime(int(ts)).tm_year

print("Loading bars...")
mid = load("cache_midcap_sip")
mid_del = load("cache_midcap_pit_delisted")
largecap = load("cache_largecap_sip")
spy = largecap["SPY"]

# ---- clean survivor universe: full 5yr grid ----
surv = {s:v for s,v in mid.items() if len(v) >= 1255}
sets = [set(int(b["timestamp"]) for b in v) for v in surv.values()]
GRID = sorted(set.intersection(*sets) & set(int(b["timestamp"]) for b in spy))
print(f"survivors(full grid)={len(surv)}  grid days={len(GRID)} "
      f"({year_of(GRID[0])}..{year_of(GRID[-1])})")
mid_del_nz = {s:v for s,v in mid_del.items() if v}
print(f"delisted(nonzero)={len(mid_del_nz)}")

# ============================================================
# PIT-aware BAB L/S (mirror of backtest_low_beta logic, entry/exit-aware)
# ============================================================
def pit_bab(bars_by_sym, bench_bars, grid, *, signal='beta', reverse=False,
            lookback=120, top_frac=0.2, rebalance_days=21, cost_bps=2.0):
    syms = sorted(bars_by_sym)
    cl = {s: {int(b["timestamp"]): float(b["close"]) for b in bars_by_sym[s]} for s in syms}
    bmap = {int(b["timestamp"]): float(b["close"]) for b in bench_bars}
    T = len(grid)
    N = len(syms)
    # price matrix on grid; nan where missing
    price = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        d = cl[s]
        for i, t in enumerate(grid):
            if t in d: price[i, j] = d[t]
    bpx = np.array([bmap.get(t, np.nan) for t in grid])
    # returns
    ret = np.full((T, N), np.nan)
    with np.errstate(all='ignore'):
        ret[1:] = price[1:] / price[:-1] - 1.0
    bret = np.zeros(T)
    bret[1:] = np.where(bpx[:-1] > 0, bpx[1:] / bpx[:-1] - 1.0, 0.0)
    # tradeable[i,j] = has price today AND yesterday (so forward return defined)
    tradeable = np.isfinite(price) & np.vstack([np.zeros((1, N), bool),
                                                np.isfinite(price[:-1])])
    k_frac = top_frac
    eq = [100000.0]; daily=[]; turns=[]
    w = np.zeros(N); prev_w = np.zeros(N)
    rebals = 0
    for t in range(1, T):
        if t > lookback and (t - 1) % rebalance_days == 0:
            win = ret[t-lookback:t]                     # (lookback, N)
            bwin = bret[t-lookback:t]
            valid = np.isfinite(win).all(axis=0) & tradeable[t]   # full window + tradeable now
            sig = np.full(N, np.nan)
            if signal == 'beta':
                var = float(np.var(bwin))
                for j in np.where(valid)[0]:
                    if var > 1e-12:
                        sig[j] = np.cov(win[:, j], bwin)[0, 1] / var
            else:
                sig[valid] = win[:, valid].std(axis=0)
            ok = np.isfinite(sig)
            nok = int(ok.sum())
            k = max(1, int(round(nok * k_frac)))
            if nok >= 2 * k and k >= 1:
                order = np.argsort(np.where(ok, sig, np.inf))
                order = order[np.isin(order, np.where(ok)[0])]
                low, high = order[:k], order[-k:]
                lng, sht = (low, high) if not reverse else (high, low)
                w = np.zeros(N)
                w[lng] = 0.5 / k
                w[sht] = -0.5 / k
                rebals += 1
        # only earn on names tradeable today; drop weight on non-tradeable (delisted) -> to cash
        wt = np.where(tradeable[t], w, 0.0)
        r_today = np.where(tradeable[t], ret[t], 0.0)
        turnover = float(np.abs(wt - prev_w).sum())
        port = float(wt @ r_today) - turnover * (cost_bps / 1e4)
        eq.append(eq[-1] * (1 + port)); daily.append(port); turns.append(turnover)
        prev_w = wt
    return dict(sharpe=sharpe_of(eq, 252.0), dd=max_drawdown_of(eq),
                ret=(eq[-1]-eq[0])/eq[0], daily=daily, dates=grid[1:],
                turn=float(np.mean(turns)) if turns else 0.0, rebals=rebals, n=len(daily))

# ============================================================
# 1) SIGN CHECK via library fn on clean survivor universe
# ============================================================
print("\n=== SIGN CHECK (library backtest_low_beta, survivor full-grid) ===")
for sig in ('beta','vol'):
    for rev in (False, True):
        r = backtest_low_beta(surv, spy, signal=sig, reverse=rev, lookback=120,
                              top_frac=0.2, rebalance_days=21, cost_bps=2.0)
        print(f"  {sig:4s} reverse={rev!s:5s}: Sharpe={r.sharpe:+.3f} ret={r.total_return:+.2%} "
              f"DD={r.max_drawdown:.2%} turn={r.avg_turnover:.3f} n={r.n_days}")

# cross-check PIT impl matches library on survivor-only
print("\n=== PIT impl vs library cross-check (survivor only, beta, reverse=False) ===")
pr = pit_bab(surv, spy, GRID, signal='beta', reverse=False)
lr = backtest_low_beta(surv, spy, signal='beta', reverse=False, lookback=120, top_frac=0.2,
                       rebalance_days=21, cost_bps=2.0)
print(f"  PIT Sharpe={pr['sharpe']:+.3f} n={pr['n']}   library Sharpe={lr.sharpe:+.3f} n={lr.n_days}")

# ============================================================
# 2) SURVIVORSHIP PIT (THE BAR) — survivor vs +delisted, entry/exit-aware
# ============================================================
print("\n=== SURVIVORSHIP PIT (PIT-aware, survivor vs +delisted) ===")
# build +delisted universe: survivors + delisted names (each trades only on its own days)
surv_plus = dict(surv); surv_plus.update(mid_del_nz)
for sig in ('beta','vol'):
    for rev in (False, True):
        s_surv = pit_bab(surv, spy, GRID, signal=sig, reverse=rev)['sharpe']
        s_plus = pit_bab(surv_plus, spy, GRID, signal=sig, reverse=rev)['sharpe']
        delta = s_plus - s_surv
        flag = "ARTIFACT (drop>0.15)->REJECT" if delta < -0.15 else "survives PIT"
        print(f"  {sig:4s} reverse={rev!s:5s}: survivor={s_surv:+.3f}  +delisted={s_plus:+.3f}  "
              f"delta={delta:+.3f}  -> {flag}")

# ============================================================
# 3) COST WALL (2/5/10 bps) on survivor universe, both signs of beta L/S
# ============================================================
print("\n=== COST WALL (beta L/S, survivor) ===")
for rev in (False, True):
    row=[]
    for c in (2.0,5.0,10.0):
        r = backtest_low_beta(surv, spy, signal='beta', reverse=rev, lookback=120,
                              top_frac=0.2, rebalance_days=21, cost_bps=c)
        row.append(f"{c:g}bps={r.sharpe:+.3f}")
    print(f"  reverse={rev!s:5s}: " + "  ".join(row))

# ============================================================
# 4) FRESH-SYMBOL HOLDOUT (i%3 split of survivor universe)
# ============================================================
print("\n=== FRESH-SYMBOL HOLDOUT (i%3) beta L/S ===")
ss = sorted(surv)
for rev in (False, True):
    parts=[]
    for grp in range(3):
        sub = {s:surv[s] for i,s in enumerate(ss) if i%3==grp}
        r = backtest_low_beta(sub, spy, signal='beta', reverse=rev, lookback=120,
                              top_frac=0.2, rebalance_days=21, cost_bps=2.0)
        parts.append(f"g{grp}={r.sharpe:+.3f}")
    print(f"  reverse={rev!s:5s}: " + "  ".join(parts))

# ============================================================
# 5) PER-YEAR + 2024+ slice (beta L/S, survivor)
# ============================================================
def per_year(daily, dates):
    by={}
    for x,t in zip(daily,dates):
        by.setdefault(year_of(t),[]).append(x)
    out={}
    for y,xs in sorted(by.items()):
        eq=[1.0]
        for x in xs: eq.append(eq[-1]*(1+x))
        out[y]=sharpe_of(eq,252.0)
    return out
print("\n=== PER-YEAR (beta L/S survivor) ===")
for rev in (False, True):
    r = pit_bab(surv, spy, GRID, signal='beta', reverse=rev)
    py = per_year(r['daily'], r['dates'])
    # 2024+ slice
    d24=[(x,t) for x,t in zip(r['daily'],r['dates']) if year_of(t)>=2024]
    eq=[1.0]; [eq.append(eq[-1]*(1+x)) for x,_ in d24]
    s24=sharpe_of(eq,252.0)
    print(f"  reverse={rev!s:5s}: " + " ".join(f"{y}:{s:+.2f}" for y,s in py.items()) + f"   |2024+:{s24:+.2f}")

# ============================================================
# 6) LEG GATE vs pairs book — does defensive low-beta diversify pairs?
# ============================================================
print("\n=== LEG GATE vs pairs book ===")
_bk = json.load(open("data/pairs_wf_returns.json"))
book_rows = _bk['returns'] if isinstance(_bk, dict) else _bk
book = {int(r['asof']): r['ret'] for r in book_rows}
print(f"pairs book: {len(book)} days, {year_of(min(book))}..{year_of(max(book))}")
def cand_dict(rev):
    r = pit_bab(surv, spy, GRID, signal='beta', reverse=rev)
    return {int(t): x for x, t in zip(r['daily'], r['dates'])}, r
for rev,label in ((False,'defensive low-beta L/S (reverse=False)'),
                  (True,'anti-defensive high-beta L/S (reverse=True)')):
    cd,_ = cand_dict(rev)
    v = evaluate_leg_candidate(cd, book, book_label='pairs')
    print(f"  {label}:")
    print(f"    passed={v.passed} rho={v.rho:+.3f} cand_sharpe={v.candidate_sharpe:+.3f} "
          f"book={v.book_sharpe:+.3f} combined={v.combined_sharpe:+.3f} lift={v.lift:+.3f} "
          f"loo+={v.loo_positive_frac:.2f} ex_recent_lift={v.ex_recent_lift:+.3f}")
    for rr in v.reasons: print("     -", rr)

# Does defensive low-beta help in pairs' 2022 hole specifically?
print("\n=== Behavior in 2022 (pairs' weak year) ===")
cd_def,_ = cand_dict(False)
for label,d in (("pairs book", book), ("defensive low-beta (rev=F)", cd_def)):
    xs=[v for t,v in d.items() if year_of(t)==2022]
    if xs:
        eq=[1.0]; [eq.append(eq[-1]*(1+x)) for x in xs]
        print(f"  {label:28s} 2022: Sharpe={sharpe_of(eq,252.0):+.3f} ret={eq[-1]-1:+.2%} n={len(xs)}")

# ============================================================
# 7) LONG-BIASED DEFENSIVE SLEEVE vs BUY-AND-HOLD
#    long-only low-beta quantile (no short). Compare to SPY B&H.
# ============================================================
print("\n=== LONG-BIASED defensive (long-only low-beta) vs SPY B&H ===")
def long_only_lowbeta(bars, bench, grid, lookback=120, top_frac=0.2, reb=21, cost=2.0, use='beta'):
    syms=sorted(bars)
    cl={s:{int(b['timestamp']):float(b['close']) for b in bars[s]} for s in syms}
    bmap={int(b['timestamp']):float(b['close']) for b in bench}
    T=len(grid); N=len(syms)
    price=np.full((T,N),np.nan)
    for j,s in enumerate(syms):
        for i,t in enumerate(grid):
            if t in cl[s]: price[i,j]=cl[s][t]
    bpx=np.array([bmap.get(t,np.nan) for t in grid])
    ret=np.full((T,N),np.nan)
    with np.errstate(all='ignore'): ret[1:]=price[1:]/price[:-1]-1.0
    bret=np.zeros(T); bret[1:]=np.where(bpx[:-1]>0,bpx[1:]/bpx[:-1]-1.0,0.0)
    trad=np.isfinite(price)&np.vstack([np.zeros((1,N),bool),np.isfinite(price[:-1])])
    eq=[1.0]; daily=[]; w=np.zeros(N); prev=np.zeros(N)
    for t in range(1,T):
        if t>lookback and (t-1)%reb==0:
            win=ret[t-lookback:t]; bwin=bret[t-lookback:t]
            valid=np.isfinite(win).all(axis=0)&trad[t]
            sig=np.full(N,np.nan); var=float(np.var(bwin))
            for j in np.where(valid)[0]:
                sig[j]=np.cov(win[:,j],bwin)[0,1]/var if var>1e-12 else np.nan
            ok=np.isfinite(sig); nok=int(ok.sum()); k=max(1,int(round(nok*top_frac)))
            if nok>=k:
                order=np.argsort(np.where(ok,sig,np.inf)); order=order[np.isin(order,np.where(ok)[0])]
                low=order[:k]; w=np.zeros(N); w[low]=1.0/k
        wt=np.where(trad[t],w,0.0); rt=np.where(trad[t],ret[t],0.0)
        turn=float(np.abs(wt-prev).sum()); port=float(wt@rt)-turn*(cost/1e4)
        eq.append(eq[-1]*(1+port)); daily.append(port); prev=wt
    return eq,daily
eq_lo,dl=long_only_lowbeta(surv,spy,GRID)
# SPY B&H over same grid
spx=np.array([ {int(b['timestamp']):float(b['close']) for b in spy}.get(t,np.nan) for t in GRID])
bh=[1.0]
for i in range(1,len(GRID)):
    bh.append(bh[-1]*(spx[i]/spx[i-1]))
print(f"  long-only low-beta: Sharpe={sharpe_of(eq_lo,252.0):+.3f} ret={eq_lo[-1]-1:+.2%} DD={max_drawdown_of(eq_lo):.2%}")
print(f"  SPY buy-and-hold  : Sharpe={sharpe_of(bh,252.0):+.3f} ret={bh[-1]-1:+.2%} DD={max_drawdown_of(bh):.2%}")
# is the sleeve POSITIVE when pairs struggle (2022)? already a long beta sleeve -> check
xs22=[x for x,t in zip(dl,GRID[1:]) if year_of(t)==2022]
eq22=[1.0]; [eq22.append(eq22[-1]*(1+x)) for x in xs22]
print(f"  long-only low-beta 2022: ret={eq22[-1]-1:+.2%} (pairs' weak year — defensive leg should be UP)")

# ============================================================
# 8) CONFIRM: reverse=True is just BETA (loses in 2022 down-year), and
#    leverage objection — beta-neutralize the low-beta leg (scale long leg up to beta-match short)
# ============================================================
print("\n=== reverse=True (anti-defensive) in 2022 down-year — is it just beta? ===")
cd_rev,_ = cand_dict(True)
xs=[v for t,v in cd_rev.items() if year_of(t)==2022]
eq=[1.0]; [eq.append(eq[-1]*(1+x)) for x in xs]
print(f"  reverse=True 2022: Sharpe={sharpe_of(eq,252.0):+.3f} ret={eq[-1]-1:+.2%}  "
      f"(if it bleeds in the down-year, the 'pass' is bull-market beta, not alpha)")
# correlation of reverse=True candidate to SPY daily
spx={int(b['timestamp']):float(b['close']) for b in spy}
spy_ret={GRID[i]:(spx[GRID[i]]/spx[GRID[i-1]]-1.0) for i in range(1,len(GRID))}
common=sorted(set(cd_rev)&set(spy_ret))
import numpy as np
a=np.array([cd_rev[t] for t in common]); b=np.array([spy_ret[t] for t in common])
print(f"  corr(reverse=True candidate, SPY) = {np.corrcoef(a,b)[0,1]:+.3f}  (high => it IS beta)")
cdf=np.array([cand_dict(False)[0][t] for t in common])
print(f"  corr(reverse=False candidate, SPY) = {np.corrcoef(cdf,b)[0,1]:+.3f}")
