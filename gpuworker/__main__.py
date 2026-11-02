"""``python -m gpuworker`` runs the CLI (same as the ``gpuworker`` console script)."""

from .cli import main

raise SystemExit(main())
