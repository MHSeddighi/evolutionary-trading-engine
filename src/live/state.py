"""Live portfolio state: JSON-persisted positions with atomic writes (crash-safe)."""
import json
import os
import tempfile
import time
from pathlib import Path

STATE_PATH = Path("results/live_state.json")


class LiveState:
    ALL_FTP = "__all__"  # cases where a rule fires while one is open on the symbol

    def __init__(self, path=STATE_PATH):
        self.path = Path(path)
        self.data = {"positions": [], "closed": [], "updated_at": None}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if not isinstance(loaded, dict) or \
                        not isinstance(loaded.get("positions", []), list):
                    raise ValueError("unexpected state schema")
                self.data = {"positions": loaded.get("positions", []),
                             "closed": loaded.get("closed", []),
                             "updated_at": loaded.get("updated_at")}
            except Exception as e:
                self.request_backup()
                print(f"⚠️  live state corrupt ({e}) — moved to .corrupt, "
                      "starting flat")

    def request_backup(self):
        try:
            os.replace(self.path, str(self.path) + ".corrupt")
        except Exception:
            pass

    # ------------ positions ------------

    def open_positions_for(self, symbol) -> list:
        return [p for p in self.data["positions"] if p["symbol"] == symbol]

    def any_open(self) -> bool:
        return bool(self.data["positions"])

    def open_position(self, **p):
        if not p.get("position_id"):
            p["position_id"] = f"{int(time.time() * 1000)}-{p['symbol']}-{len(self.data['positions'])}"
        p["opened_at"] = time.time()
        self.data["positions"].append(p)
        self.save()
        return p

    def update_position(self, pos: dict):
        """Persist in-place position progress (bars held / last checked bar)."""
        for i, p in enumerate(self.data["positions"]):
            if p["position_id"] == pos["position_id"]:
                self.data["positions"][i] = pos
                self.save()
                return pos
        raise KeyError(f"position {pos['position_id']} not open")

    def close_position(self, position_id: str, exit_record: dict):
        pos, rest = None, []
        for p in self.data["positions"]:
            if p["position_id"] == position_id:
                pos = p
            else:
                rest.append(p)
        if pos is None:
            raise KeyError(f"position {position_id} not open")
        exit_rec = {**pos, **exit_record, "closed_at": time.time()}
        self.data["positions"] = rest
        self.data["closed"].append(exit_rec)
        self.save()
        return exit_rec

    # ------------ persistence ------------

    def save(self):
        self.data["updated_at"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = tempfile.NamedTemporaryFile("w", delete=False, dir=self.path.parent)
        json.dump(self.data, fd, indent=1)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.replace(fd.name, self.path)
