"""Verify the saved study without drawing observations or changing its files."""

import hashlib
import json
from pathlib import Path

import numpy as np

from run_adaptive_validation import recheck_archive


HERE = Path(__file__).resolve().parent
OUT = HERE / "adaptive_results"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (OUT / "source_manifest_before.json").read_text(encoding="utf-8")
    )
    current = {name: sha(HERE / name) for name in manifest}
    assert current == manifest == data["source_hashes_before"]
    assert current == data["source_hashes_after"]
    protocol = json.loads((HERE / "adaptive_protocol.json").read_text(encoding="utf-8"))
    assert protocol == data["protocol"]
    archive = OUT / "trial_records.npz"
    assert sha(archive) == data["archive_sha256"]
    checked = recheck_archive(archive, data["results"])
    assert checked == data["archive_recheck"]
    total = 0
    with np.load(archive, allow_pickle=False) as records:
        for rec in data["results"]:
            cid, method, budget = rec["id"], rec["method"], rec["B"]
            prefix = f"{cid}_{method}"
            pilot_cost = 0 if method == "known_a" else 2 * (budget // 8)
            used = records[prefix + "_used_bits"]
            if method == "dyadic":
                cost = pilot_cost + sum(2 * x["r"] * x["n"] for x in rec["schedule"])
                assert np.all(used == cost)
            else:
                depths = records[prefix + "_depths"]
                counts = records[prefix + "_fresh_pairs"]
                expected = pilot_cost + depths.sum(axis=1) * counts
                assert np.array_equal(used, expected)
                assert np.all(depths >= 1) and np.all(counts >= 1)
            assert np.all(used <= budget)
            assert rec["used_bits_min"] == int(used.min())
            assert rec["used_bits_max"] == int(used.max())
            estimates = records[prefix + "_rho"]
            assert len(estimates) == protocol["trials"]
            assert np.isfinite(estimates).all() and np.all(np.abs(estimates) <= 1)
            normalized = budget * min(rec["a1"], rec["a2"]) ** 2 * rec["mse"]
            assert normalized == rec["normalized_mse"]
            total += len(estimates)
        arrays = len(records.files)
    assert total == data["saved_estimate_count"] == 510000
    assert len(data["results"]) == data["method_case_count"] == 255
    assert len({r["id"] for r in data["results"]}) == data["configuration_count"] == 51
    return {
        "status": "pass",
        "scope": "saved-source identity, estimator reconstruction, summaries, budgets",
        "source_files": len(current),
        "stored_arrays": arrays,
        "saved_estimates": total,
        **checked,
        "new_random_observations": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
