"""Overnight queue: for each candidate encoder, extract frozen features for CBIS-DDSM and
Mini-DDSM (resumable; skips images already embedded), then run encoder_eval.py.
Each step runs in its own process so GPU memory is released between encoders.
Run: python overnight_queue.py enc1 enc2 ...   (log: cache/overnight_queue.log)"""
import subprocess
import sys
import time

DATASETS = ["cbis_full", "miniddsm", "miniddsm_normal"]


def run(cmd, log):
    t = time.time()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t


def main():
    with open("cache/overnight_queue.log", "a") as log:
        for enc in sys.argv[1:]:
            ok = True
            for ds in DATASETS:
                code, dt = run([sys.executable, "-c",
                                f"import embed_all; embed_all.embed_all('{enc}', dataset='{ds}')"], log)
                print(f"{time.strftime('%H:%M')} {enc} embed {ds}: exit {code} ({dt:.0f}s)", flush=True)
                ok &= code == 0
            if not ok:
                print(f"{time.strftime('%H:%M')} {enc}: SKIPPED eval (embedding failed)", flush=True)
                continue
            code, dt = run([sys.executable, "encoder_eval.py", enc], log)
            print(f"{time.strftime('%H:%M')} {enc} eval: exit {code} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
