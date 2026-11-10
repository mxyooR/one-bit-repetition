"""Run the registered unknown-reliability study once and retain trial records."""

from pathlib import Path
import hashlib
import itertools
import json
import math
import platform
import sys
import time
import numpy as np
import scipy

from adaptive_model import (adaptive_depths, cross_law, draw_cross_means,
                            dyadic_schedule, gamma, pilot_estimates,
                            rho_from_raw, summarize)

HERE = Path(__file__).resolve().parent
OUT = HERE / "adaptive_results"
PROTOCOL = HERE / "adaptive_protocol.json"
SOURCES = [HERE / "adaptive_model.py", Path(__file__).resolve(), PROTOCOL]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_checks():
    records = []
    for (r1, r2), (a1, a2), q in itertools.product(
        [(1, 1), (2, 3), (3, 2), (5, 7), (16, 8)],
        [(.05, .2), (.3, .8), (1., .4), (1., 1.)],
        [-.7, 0., .65],
    ):
        p, v = cross_law(r1, r2, a1, a2, q)
        raw_mean = a1 * a2 * q
        raw_second = (a1*a1 + (1-a1*a1)/r1) * (a2*a2 + (1-a2*a2)/r2)
        pp, vv = cross_law(r1, r2, a1, a2, q, True)
        gg = gamma(r1, a1) * gamma(r2, a2)
        errs = [abs(p.sum()-1), abs(p @ v-raw_mean),
                abs(p @ (v*v)-raw_second),
                abs(pp @ vv-q*gg), abs(pp @ (vv*vv)-gg)]
        assert max(errs) < 1e-10, (r1, r2, a1, a2, q, errs)
        records.append({"r1":r1, "r2":r2, "a1":a1, "a2":a2,
                        "q":q, "max_abs_error":max(errs)})
    # Enumerate latent signs and every flip at unequal depths, independently
    # of the count law's sign-invariance reduction.
    r1, r2, a1, a2, q = 2, 3, .3, .8, -.4
    law = {}
    for s1, s2 in itertools.product([-1, 1], repeat=2):
        for flips in itertools.product([-1, 1], repeat=r1+r2):
            p = (1+q*s1*s2)/4
            for i, x in enumerate(flips):
                p *= (1+(a1 if i < r1 else a2)*x)/2
            numerator = s1*s2*sum(flips[:r1])*sum(flips[r1:])
            law[numerator] = law.get(numerator, 0.) + p
    p, v = cross_law(r1, r2, a1, a2, q)
    reduced = {}
    for prob, value in zip(p, v):
        key = int(round(value*r1*r2))
        reduced[key] = reduced.get(key, 0.) + prob
    error = max(abs(law.get(k,0)-reduced.get(k,0))
                for k in set(law)|set(reduced))
    assert error < 1e-12
    budgets = [16, 17, 31, 64, 101, 16384, 262144, 4194304]
    for B in budgets:
        m = B//8
        h = np.array([[1/math.sqrt(m),1/math.sqrt(m)],
                      [1/math.sqrt(m),1.], [1.,1.]])
        rr = adaptive_depths(h, m)
        nn = (B-2*m)//rr.sum(axis=1)
        assert np.all(rr <= m) and np.all(nn >= 3)
        assert np.all(2*m+nn*rr.sum(axis=1) <= B)
        sch = dyadic_schedule(B)
        assert all(r >= 1 and n >= 1 for r,n in sch)
        assert 2*m+sum(2*r*n for r,n in sch) == B-B%2
    return {"moment_cases":records, "enumerated_law_max_error":error,
            "engineering_budget_cases":budgets, "tolerance":1e-10}


def run_configuration(cfg, trials, seed, arrays):
    B, a1, a2, rho = cfg["B"], cfg["a1"], cfg["a2"], cfg["rho"]
    q = 2/math.pi * math.asin(rho)
    m = B//8
    methods = ["two_stage","dyadic","fixed_2","fixed_100","known_a"]
    streams = np.random.SeedSequence(seed).spawn(1+len(methods))
    pilot_rng = np.random.default_rng(streams[0])
    s1 = 2*pilot_rng.binomial(1,.5,size=trials)-1
    s2 = s1*(2*pilot_rng.binomial(1,(1+q)/2,size=trials)-1)
    sums = np.column_stack([
        s1*(2*pilot_rng.binomial(m,(1+a1)/2,size=trials)-m),
        s2*(2*pilot_rng.binomial(m,(1+a2)/2,size=trials)-m),
    ])
    ahat = pilot_estimates(sums,m)
    cid = cfg["id"]
    arrays[cid+"_pilot_sums"] = sums
    arrays[cid+"_ahat"] = ahat
    arrays[cid+"_pilot_floor"] = np.abs(sums)/m < 1/math.sqrt(m)
    results = []
    for mi, method in enumerate(methods):
        rng = np.random.default_rng(streams[mi+1])
        prefix = cid+"_"+method
        record = dict(cfg, method=method,
                      seed_state=streams[mi+1].generate_state(4).tolist())
        if method == "dyadic":
            schedule = dyadic_schedule(B)
            means = np.column_stack([
                draw_cross_means(rng,r,r,a1,a2,q,n,trials)
                for r,n in schedule])
            depths = np.array([r for r,n in schedule])
            ns = np.array([n for r,n in schedule])
            v = np.prod(1+(ahat[:,:,None]**-2-1)/depths,axis=1)
            weights = ns/v
            raw = np.sum(weights*means,axis=1)/weights.sum(axis=1)/ahat.prod(axis=1)
            used = np.full(trials,2*m+sum(2*r*n for r,n in schedule),dtype=np.int64)
            arrays[prefix+"_cross_means"] = means
            record["schedule"] = [{"r":r,"n":n} for r,n in schedule]
        else:
            if method == "two_stage":
                rr = adaptive_depths(ahat,m)
                ns = (B-2*m)//rr.sum(axis=1)
                den = ahat.prod(axis=1)
                pilot_cost = 2*m
            elif method == "known_a":
                rr = np.tile([math.ceil(a1**-2),math.ceil(a2**-2)],(trials,1))
                ns = B//rr.sum(axis=1)
                den = gamma(int(rr[0,0]),a1)*gamma(int(rr[0,1]),a2)
                pilot_cost = 0
                record["posterior_product_factor"] = float(den)
            else:
                depth = 2 if method == "fixed_2" else 100
                rr = np.full((trials,2),depth,dtype=np.int64)
                ns = (B-2*m)//rr.sum(axis=1)
                den = ahat.prod(axis=1)
                pilot_cost = 2*m
            means = np.empty(trials)
            for r1,r2 in np.unique(rr,axis=0):
                ix = np.flatnonzero((rr[:,0]==r1)&(rr[:,1]==r2))
                n = int(ns[ix[0]])
                means[ix] = draw_cross_means(
                    rng,int(r1),int(r2),a1,a2,q,n,len(ix),method=="known_a")
            raw = means/den
            used = pilot_cost+ns*rr.sum(axis=1)
            arrays[prefix+"_cross_means"] = means
            arrays[prefix+"_depths"] = rr
            arrays[prefix+"_fresh_pairs"] = ns
            record["depth1_median"] = float(np.median(rr[:,0]))
            record["depth2_median"] = float(np.median(rr[:,1]))
            record["fresh_pairs_median"] = float(np.median(ns))
        est = rho_from_raw(raw)
        arrays[prefix+"_raw_q"] = raw
        arrays[prefix+"_rho"] = est
        arrays[prefix+"_used_bits"] = used
        assert np.all(used <= B) and np.isfinite(est).all()
        record.update(summarize(est,rho))
        record["normalized_mse"] = B*min(a1,a2)**2*record["mse"]
        record["used_bits_min"] = int(used.min())
        record["used_bits_max"] = int(used.max())
        record["pilot_floor_frequency"] = (
            None if method=="known_a" else
            float(np.any(arrays[cid+"_pilot_floor"],axis=1).mean()))
        results.append(record)
    return results


def recheck_archive(path, results):
    max_stat_diff = 0.
    max_estimate_diff = 0.
    with np.load(path) as data:
        for rec in results:
            cid, method = rec["id"], rec["method"]
            prefix = cid+"_"+method
            ahat = pilot_estimates(data[cid+"_pilot_sums"],rec["B"]//8)
            means = data[prefix+"_cross_means"]
            if method == "dyadic":
                r = np.array([s["r"] for s in rec["schedule"]])
                n = np.array([s["n"] for s in rec["schedule"]])
                weights = n/np.prod(1+(ahat[:,:,None]**-2-1)/r,axis=1)
                raw = (weights*means).sum(axis=1)/weights.sum(axis=1)/ahat.prod(axis=1)
            elif method == "known_a":
                rr = data[prefix+"_depths"][0]
                raw = means/(gamma(int(rr[0]),rec["a1"])*gamma(int(rr[1]),rec["a2"]))
            else:
                raw = means/ahat.prod(axis=1)
            estimates = rho_from_raw(raw)
            max_estimate_diff = max(max_estimate_diff,float(
                np.max(np.abs(estimates-data[prefix+"_rho"]))))
            stats = summarize(estimates,rec["rho"])
            max_stat_diff = max(max_stat_diff,max(abs(stats[k]-rec[k]) for k in stats))
    assert max_estimate_diff == 0 and max_stat_diff == 0
    return {"max_estimate_difference":max_estimate_diff,
            "max_summary_difference":max_stat_diff,
            "parameter_method_cases":len(results)}


def main():
    if OUT.exists():
        raise SystemExit("Output directory exists; preserve the registered run.")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    before = {p.name:sha(p) for p in SOURCES}
    OUT.mkdir()
    (OUT/"source_manifest_before.json").write_text(
        json.dumps(before,indent=2)+"\n",encoding="utf-8")
    started = time.perf_counter()
    checks = exact_checks()
    configurations = []
    for B,a,rho in itertools.product(protocol["budgets"],
                                    protocol["equal_reliabilities"],
                                    protocol["correlations"]):
        configurations.append({"B":B,"a1":a,"a2":a,"rho":rho,"family":"equal"})
    configurations.extend(dict(c,family="unequal") for c in protocol["unequal_cases"])
    arrays,results = {},[]
    seeds = np.random.SeedSequence(protocol["root_seed"]).spawn(len(configurations))
    for i,(cfg,ss) in enumerate(zip(configurations,seeds)):
        cfg = dict(cfg,id=f"c{i:03d}")
        seed = ss.generate_state(4).tolist()
        cfg["configuration_seed"] = seed
        results.extend(run_configuration(cfg,protocol["trials"],seed,arrays))
        print(f"Completed {i+1}/{len(configurations)} configurations",flush=True)
    archive = OUT/"trial_records.npz"
    np.savez_compressed(archive,**arrays)
    recheck = recheck_archive(archive,results)
    after = {p.name:sha(p) for p in SOURCES}
    assert before == after
    payload = {
        "protocol":protocol,
        "configuration_count":len(configurations),
        "method_case_count":len(results),
        "saved_estimate_count":len(results)*protocol["trials"],
        "elapsed_seconds":time.perf_counter()-started,
        "environment":{"python":sys.version.split()[0],"numpy":np.__version__,
                       "scipy":scipy.__version__,"os":platform.system()},
        "source_hashes_before":before,"source_hashes_after":after,
        "archive_sha256":sha(archive),
        "archive_recheck":recheck,"results":results,
    }
    (OUT/"exact_checks.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    (OUT/"results.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:payload[k] for k in [
        "configuration_count","method_case_count","saved_estimate_count",
        "elapsed_seconds","archive_recheck"]}),flush=True)


if __name__ == "__main__":
    main()
