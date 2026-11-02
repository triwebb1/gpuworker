"""Oneshot echo worker. The broker sets GPUWORKER_BUNDLE / GPUWORKER_OUT /
GPUWORKER_JOB_ID (in a container the bundle and outputs are mounted at /job/bundle and
/job/out). Reads the job manifest, writes declared outputs."""

import json
import os
import sys


def main() -> int:
    job_id = os.environ.get("GPUWORKER_JOB_ID", "")
    bundle_dir = os.environ.get("GPUWORKER_BUNDLE", "/job/bundle")
    out_dir = os.environ.get("GPUWORKER_OUT", "/job/out")
    with open(
        os.path.join(bundle_dir, "jobs", "pending", f"{job_id}.json"), encoding="utf-8"
    ) as fh:
        manifest = json.load(fh)
    outputs = manifest.get("outputs") or [f"out/{job_id}.txt"]
    text = manifest.get("options", {}).get("text", f"echo {job_id}")
    # Write to both streams so the smoke test can check the worker log the broker packs
    # into the result bundle (_gpuworker/<job_id>.log); an empty log would be
    # indistinguishable from a broken one.
    print(f"ECHO-STDOUT job_id={job_id} outputs={len(outputs)}")
    print(f"ECHO-STDERR job_id={job_id} text={text!r}", file=sys.stderr)
    root = os.path.realpath(out_dir)
    for rel in outputs:
        dest = os.path.realpath(os.path.join(root, rel))
        if os.path.isabs(rel) or not dest.startswith(root + os.sep):
            # The broker refuses such a manifest; a worker still never writes outside its
            # output directory on the manifest's say-so.
            print(f"refusing output outside the output directory: {rel!r}", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
