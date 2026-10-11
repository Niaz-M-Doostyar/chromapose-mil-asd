
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "experiments/queue_improvement.txt"


def main():
    active = subprocess.run(
        ["/usr/bin/pgrep", "-fl", "train.py|run_queue.sh"],
        capture_output=True, text=True, check=False,
    )
    if active.returncode not in (0, 1):
        raise RuntimeError(f"Cannot verify queue activity: {active.stderr}")
    if active.stdout.strip():
        raise RuntimeError(f"A research process is already active:\n{active.stdout}")
    stamp = datetime.datetime.now(datetime.timezone.utc)
    audit = ROOT / "results/audit"
    audit.mkdir(parents=True, exist_ok=True)
    with (ROOT / "results/improvement_queue.log").open("a") as log:
        process = subprocess.Popen(
            ["/usr/bin/nohup", "/usr/bin/caffeinate", "-i", "/bin/zsh",
             str(ROOT / "run_queue.sh"), str(QUEUE)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    record = dict(
        started_utc=stamp.isoformat(), pid=process.pid,
        queue=str(QUEUE), queue_sha256=hashlib.sha256(QUEUE.read_bytes()).hexdigest(),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        log=str(ROOT / "results/improvement_queue.log"),
        behavior="One sequential GPU job; completed prediction folds skipped. "
                 "An unfinished fold retrains from scratch. Detached session and "
                 "SIGHUP protection; machine shutdown still interrupts execution.",
    )
    path = audit / f"queue_launch_{stamp.strftime('%Y-%m-%d_%H%M%S')}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
