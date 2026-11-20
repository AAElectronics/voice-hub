"""On-board speaker recognition for LocalDot boards (development firmware, ESP32-P4).

The board computes its own voiceprint of every command (TitaNet small, on the P4) and
fires the Home Assistant event `esphome.localdot_speaker`:
    {device_id, name, score, detail, source: "live"|"clip", voiceprint: 768 hex chars}
(192 int16 values, cosine × 1000). Voice Hub:

  1. attaches that voiceprint to the command it captured from the same satellite
     (field `p4` in the recording's .json), so the board's guess shows next to ours;
  2. when people change (a command labelled with "Who spoke?", a sample removed, a
     person removed), builds the board's people list from the recordings that carry a
     board voiceprint and sends it in pieces to every board that has the actions
     `p4_people_begin` / `p4_people_chunk` / `p4_people_finish`.

The board voiceprints live in another embedding space than Voice Hub's, so they are
never mixed: the board only gets voiceprints the board itself computed.

People list format ("SPP1", little endian): magic | n u16 | 192 u16 | threshold f32 |
margin f32 | n × (name 24 bytes, NUL padded | int8[192] = unit voiceprint × 127).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import time
import zlib

import numpy as np

_LOG = logging.getLogger("voice-hub.p4")

EVENT = "esphome.localdot_speaker"
ATTACH_S = 20.0          # the board's event and Voice Hub's command must be this close
MAX_ENTRIES = 75         # 16 kB on the board
PER_PERSON = 20          # newest samples per person when there are too many
THRESHOLD, MARGIN = 0.50, 0.10   # measured 2026-11-17 on board voiceprints (same person ≥ 0.56, others ≤ 0.53)
CHUNK = 3000


def unit(hexstr: str) -> np.ndarray | None:
    if not isinstance(hexstr, str) or len(hexstr) != 768:
        return None
    try:
        v = np.array([int(hexstr[i:i + 4], 16) for i in range(0, 768, 4)], dtype=np.int32)
    except ValueError:
        return None
    v[v >= 32768] -= 65536
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def people_list(people: dict[str, list[str]], threshold: float = THRESHOLD, margin: float = MARGIN) -> bytes:
    """people: name -> [voiceprint hex, ...] (oldest first)."""
    entries = []
    keep = max(1, min(PER_PERSON, MAX_ENTRIES // max(1, len(people))))
    for name, prints in sorted(people.items()):
        for h in prints[-keep:]:
            v = unit(h)
            if v is None:
                continue
            q = np.clip(np.round(v * 127), -127, 127).astype(np.int8)
            entries.append(name.encode()[:24].ljust(24, b"\0") + q.tobytes())
    return b"SPP1" + struct.pack("<HHff", len(entries), 192, threshold, margin) + b"".join(entries)


class P4Speaker:
    def __init__(self, hub) -> None:
        self.hub = hub
        self.pending: dict[str, tuple[float, dict]] = {}   # device_id -> (time, event data) not attached yet
        self.last_sync: dict[str, str] = {}                # device_id -> crc of the list it got
        self.status: dict[str, str] = {}                   # device_id -> readable status for the page
        self._sync_task: asyncio.Task | None = None

    # ═══ 1. board voiceprint → command ═══
    async def subscribe(self, capture) -> None:
        self.capture = capture
        await capture.ha.call({"type": "subscribe_events", "event_type": EVENT}, on_event=self._on_event)
        self.schedule_sync()      # boards get the list at start (only if it changed since)

    def _on_event(self, event: dict) -> None:
        d = event.get("data") or {}
        if "no people enrolled" in str(d.get("detail", "")) and d.get("device_id"):
            self.last_sync.pop(d["device_id"], None)    # the board lost its list (new firmware?): send it again
            self.schedule_sync()
        if d.get("source") != "live" or unit(d.get("voiceprint")) is None or not d.get("device_id"):
            return
        p4 = {"person": d.get("name") or "?", "score": _num(d.get("score")), "detail": d.get("detail", ""),
              "voiceprint": d["voiceprint"]}
        if not self._attach(d["device_id"], p4):
            self.pending[d["device_id"]] = (time.time(), p4)

    def on_command(self, entry: dict, device_id: str | None) -> None:
        """Called when Voice Hub adds a command: the board's event may have come first."""
        got = self.pending.get(device_id or "")
        if got and time.time() - got[0] < ATTACH_S:
            self.pending.pop(device_id)
            self._attach(device_id, got[1], entry)

    def _attach(self, device_id: str, p4: dict, entry: dict | None = None) -> bool:
        if entry is None:
            now = time.time()
            for e in reversed(self.hub.recent):
                if e.get("device_id") != device_id or "p4" in e:
                    continue
                t = time.mktime(time.strptime(e["time"], "%Y-%m-%dT%H:%M:%S"))
                if now - t < ATTACH_S:
                    entry = e
                break
        if entry is None:
            return False
        entry["p4"] = {k: v for k, v in p4.items() if k != "voiceprint"}
        if entry.get("file"):
            self.hub.update_collected(os.path.join(self.hub.collection, entry["file"]), {"p4": p4})
        _LOG.info("board voiceprint attached to %s: %s %.2f", entry.get("file"), p4["person"], p4["score"])
        return True

    # ═══ 2. people → board ═══
    def board_people(self) -> dict[str, list[str]]:
        reg = self.hub.registry
        reg.reload_if_changed()
        out: dict[str, list[str]] = {}
        for name, files in reg.sources.items():
            for f in files:
                try:
                    with open(os.path.join(self.hub.collection, f + ".json")) as fh:
                        h = (json.load(fh).get("p4") or {}).get("voiceprint")
                except (OSError, ValueError):
                    continue
                if unit(h) is not None:
                    out.setdefault(name, []).append(h)
        return out

    def schedule_sync(self) -> None:
        """People changed: send the list a moment later (several edits in a row → one transfer)."""
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()
        self._sync_task = asyncio.create_task(self._sync_later())

    async def _sync_later(self) -> None:
        await asyncio.sleep(3)
        try:
            await self.sync()
        except Exception as e:
            _LOG.warning("board people list: %s", e)

    async def boards(self) -> dict[str, str]:
        """device_id -> ESPHome service prefix, for boards with the p4_people actions."""
        async with self.hub.session.get(f"{self.hub.ha_url}/api/services", headers=self.hub.headers()) as r:
            services = await r.json()
        esphome = next((s.get("services", {}) for s in services if s.get("domain") == "esphome"), {})
        prefixes = {k[:-len("_p4_people_begin")] for k in esphome if k.endswith("_p4_people_begin")}
        nodes = dict(getattr(self.hub, "board_nodes", {}))
        cap = getattr(self, "capture", None)
        if cap is not None:          # at start nobody opened the page yet: read the node names ourselves
            import ring
            async with self.hub.session.get(f"{self.hub.ha_url}/api/states", headers=self.hub.headers()) as r:
                states = {s["entity_id"]: s for s in await r.json()}
            for device_id, ents in cap.device_entities.items():
                node = ring.node_of(ents, states)
                if node:
                    nodes[device_id] = node
        out = {}
        for device_id, node in nodes.items():
            p = node.replace("-", "_")
            if p in prefixes:
                out[device_id] = p
        return out

    async def sync(self, force: bool = False) -> None:
        blob = people_list(self.board_people())
        crc = f"{zlib.crc32(blob) & 0xffffffff:08x}"
        n = struct.unpack("<H", blob[4:6])[0]
        for device_id, prefix in (await self.boards()).items():
            if not force and self.last_sync.get(device_id) == crc:
                continue
            base = f"{self.hub.ha_url}/api/services/esphome/{prefix}_p4_people_"
            calls = [("begin", {"size": len(blob), "crc32": crc})]
            calls += [("chunk", {"offset": o, "data": base64.b64encode(blob[o:o + CHUNK]).decode()})
                      for o in range(0, len(blob), CHUNK)]
            calls.append(("finish", {}))
            try:
                for action, data in calls:
                    async with self.hub.session.post(base + action, headers=self.hub.headers(), json=data) as r:
                        if r.status >= 300:
                            raise RuntimeError(f"{action}: {r.status} {await r.text()}")
            except Exception as e:
                self.status[device_id] = f"error: {e}"
                _LOG.warning("board people list to %s: %s", prefix, e)
                continue
            self.last_sync[device_id] = crc
            self.status[device_id] = f"{n} voiceprints sent {time.strftime('%H:%M')}"
            _LOG.info("board people list to %s: %d voiceprints, %d bytes, CRC32 %s", prefix, n, len(blob), crc)

    def summary(self) -> dict:
        return {"people": {k: len(v) for k, v in self.board_people().items()}, "status": self.status}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
