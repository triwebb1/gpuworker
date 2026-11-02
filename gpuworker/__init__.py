"""gpuworker: a broker that owns one GPU and arbitrates it across several model
services. See SPEC.md (design) and docs/MODULE_GUIDE.md (module-author contract).

The package is layered so the deterministic core (transport, manifests, registry,
scheduler admission) is testable on any host, and the host-specific parts (NVML,
container launch) sit behind abstractions with mock backends.
"""

__version__ = "0.1.0"

# Broker <-> serve-container wire protocol version (SPEC 4.3). Bump when the
# frame schema in serve.py changes incompatibly.
PROTOCOL_VERSION = 1
