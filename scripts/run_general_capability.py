"""General-capability evaluation (LM Evaluation Harness).

CAPE is weight-preserving and capability queries are out of scope (the gate rejects them),
so CAPE's general capability equals the unedited base model's. This measures that base
level on the same four benchmarks the comparison table reports.

Requires `lm-eval==0.4.5` and a hub token in HF_TOKEN.
"""
import glob
import json
import os
import subprocess
import sys

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
TASKS = ("mmlu", "commonsense_qa", "anli_r1", "triviaqa")
LIMIT = int(os.environ.get("LIMIT", "200"))
BATCH_SIZE = os.environ.get("BATCH_SIZE", "8")
OUTDIR = os.path.join(os.environ.get("WORK_DIR", "./.run"), "capability")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={MODEL},dtype=bfloat16",
        "--tasks", ",".join(TASKS),
        "--limit", str(LIMIT),
        "--batch_size", BATCH_SIZE,
        "--output_path", OUTDIR,
    ]
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

    for f in sorted(glob.glob(OUTDIR + "/**/results*.json", recursive=True)):
        res = json.load(open(f)).get("results", {})
        for task in TASKS:
            if task in res:
                m = {k: round(v, 4) for k, v in res[task].items()
                     if isinstance(v, float) and ("acc" in k or "exact_match" in k or "f1" in k)}
                print("  %-16s %s" % (task, m), flush=True)


if __name__ == "__main__":
    main()
