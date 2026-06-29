"""ingest_gate node (§1 ingest gate).

Hard constraint: before any processing, rights must be explicitly confirmed.
Aborts if absent. Logs URL + timestamp + confirmation to runs/<id>/rights.json.
The actual y/N prompt happens in the CLI; this node enforces+records the result.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from ..config import run_path
from ..logging_utils import log_event
from ..types import ClipState


class RightsNotConfirmedError(RuntimeError):
    pass


def ingest_gate(state: ClipState) -> dict:
    run_id = state["run_id"]
    url = state["source_url"]
    confirmed = bool(state.get("rights_confirmed"))

    if not confirmed:
        log_event("ingest_gate", "rights_not_confirmed", level="error", url=url)
        raise RightsNotConfirmedError(
            "Rights not confirmed. Re-run with --i-have-rights (or answer 'y' to the "
            "confirmation prompt). See README §Operator responsibility."
        )

    record = {
        "run_id": run_id,
        "source_url": url,
        "rights_confirmed": True,
        "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
    }
    rights_file = run_path(run_id) / "rights.json"
    rights_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    log_event("ingest_gate", "rights_confirmed", url=url, file=str(rights_file))
    return {"rights_confirmed": True}
