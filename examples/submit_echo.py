"""Submit one echo.echo job to a queue tree and print the result.

The broker is a separate process (`gpuworker --config examples/local.toml --mock-gpu
serve`); this script only touches the queue directory, the same way any client does.

    python examples/submit_echo.py ./queues/demo "hello"
"""

import sys
import uuid

from gpuworker.client import collect, submit
from gpuworker.manifest import JobManifest


def main() -> int:
    queue_root = sys.argv[1] if len(sys.argv) > 1 else "./queues/demo"
    text = sys.argv[2] if len(sys.argv) > 2 else "hello from echo"
    job_id = f"echo-{uuid.uuid4().hex[:8]}"
    manifest = JobManifest.from_dict(
        {
            "job_id": job_id,
            "job_type": "echo.echo",
            "outputs": ["out/hello.txt"],
            "options": {"text": text},
        }
    )
    bundle = submit(queue_root, [manifest])
    print(f"submitted {bundle} to {queue_root}/inbox")
    result = collect(queue_root, bundle, dest_root=f"outputs/{job_id}", timeout_s=60)[0]
    print(f"status : {result.status}")
    print(f"message: {result.message}")
    for rel in (result.result or {}).get("outputs", []):
        with open(f"outputs/{job_id}/{rel}", encoding="utf-8") as fh:
            print(f"output : outputs/{job_id}/{rel} -> {fh.read().strip()!r}")
    return 0 if result.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
