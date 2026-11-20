"""Voice Hub as a conversation agent: who spoke is known BEFORE the command runs.

Home Assistant's own Wyoming integration accepts a conversation agent (a
Wyoming "handle" service). With Voice Hub chosen as the pipeline's agent:

  1. Home Assistant sends the transcript, with the device and satellite that
     heard it;
  2. Voice Hub closes that command's voiceprint at once (the audio is over:
     no need to wait for the stream gap) and waits for it, at most WAIT_S;
  3. it publishes who spoke (event + sensor, as always), then passes the text
     to the real agent - Home Assistant, OpenAI, Google, ... chosen on the page,
     per device or per person - with the same device, so rooms still work;
  4. the reply goes back to the satellite.

Follow-up questions: an LLM agent asks Home Assistant to keep listening when
its reply ends with a question mark, but a Wyoming agent can only return the
reply's text. Voice Hub reads that flag from the real agent, waits for the
satellite to finish speaking and reopens listening with
`assist_satellite.start_conversation`; the next sentence from that satellite
goes on in the same conversation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

from wyoming.asr import Transcript
from wyoming.handle import Handled, NotHandled
from wyoming.info import Attribution, HandleModel, HandleProgram

_LOG = logging.getLogger("voice-hub.agent")

WAIT_S = 1.2              # longest wait for the voiceprint before the command goes on
FOLLOW_UP_S = 25.0        # a reopened listen counts as the same conversation this long
IDLE_TIMEOUT_S = 60.0     # longest wait for the satellite to finish the reply
DEFAULT_AGENT = "conversation.home_assistant"

# Timer phrasings Home Assistant's built-in Italian sentences miss, rewritten to one it knows
# ("imposta un timer di ..."). Speech-to-text often hears "imposto" for "imposta"; "metti un timer"
# even matched media playback. Only when a duration follows, so other commands are untouched.
_DURATION = re.compile(r"\b(second[oi]|minut[oi]|or[ae]|mezz)", re.I)
_TIMER_IT = [
    re.compile(r"^(?:mi\s+)?(?:imposto|impostami|imposta|metti|mettimi|fammi|fai|setta|settami|crea|creami|"
               r"avvia|avviami|fai\s+partire|programma)\s+(?:un|il|uno)\s+timer\s+(?:di|da|per|a|su)\s+(.+)$", re.I),
    re.compile(r"^(?:un\s+)?timer\s+(?:di|per|da|a)\s+(.+)$", re.I),
    re.compile(r"^(.+?)\s+di\s+timer$", re.I),
]
# Durations in words that Home Assistant's sentences do not parse ("un'ora" fails even in the right phrase).
_WORDS_IT = [
    (re.compile(r"\bun\s+quarto\s+d['’\s]?ora\b", re.I), "15 minuti"),
    (re.compile(r"\bmezz['’\s]?ora\b|\bmezza\s+ora\b", re.I), "30 minuti"),
    (re.compile(r"\b(?:un|una)\s*['’]?\s*ora\b", re.I), "1 ora"),
    (re.compile(r"\bun\s+minuto\b", re.I), "1 minuto"),
    (re.compile(r"\bun\s+secondo\b", re.I), "1 secondo"),
    (re.compile(r"\b(or[ae])\s+e\s+mezz[ao]\b", re.I), r"\1 e 30 minuti"),
    (re.compile(r"\b(minut[oi])\s+e\s+mezz[ao]\b", re.I), r"\1 e 30 secondi"),
]


def canonical_timer(text: str) -> str | None:
    """The phrase Home Assistant recognises for an Italian timer request, or None if not one."""
    t = text.strip().rstrip(".!?").strip()
    for rx in _TIMER_IT:
        m = rx.match(t)
        if m and _DURATION.search(m.group(1)):
            d = m.group(1).strip()
            for w, n in _WORDS_IT:
                d = w.sub(n, d)
            return f"imposta un timer di {d}"
    return None


class Agent:
    """Settings, conversation threads and the forwarding itself."""

    def __init__(self, hub, capture, data: str) -> None:
        self.hub, self.capture = hub, capture
        self.path = os.path.join(data, "agent.json")
        self.settings = {"default": DEFAULT_AGENT, "devices": {}, "people": {}, "tell_name": True}
        try:
            self.settings.update(json.load(open(self.path)))
        except Exception:
            pass
        self.threads: dict[str, str] = {}               # Home Assistant conversation id -> real agent's
        self.follow_up: dict[str, tuple[float, str]] = {}  # device_id -> (until, real agent's conversation id)
        self.last: list[dict] = []                      # recent turns, for the page
        self._passing: set[tuple] = set()               # (device, text) being passed on: catches a loop to itself

    def save(self, new: dict) -> dict:
        s = dict(self.settings)
        if "default" in new and str(new["default"]).startswith("conversation."):
            s["default"] = new["default"]
        for key in ("devices", "people"):
            if isinstance(new.get(key), dict):
                s[key] = {k: v for k, v in new[key].items() if isinstance(v, str) and v.startswith("conversation.")}
        if "tell_name" in new:
            s["tell_name"] = bool(new["tell_name"])
        self.settings = s
        with open(self.path, "w") as f:
            json.dump(s, f, indent=1)
        return s

    def info(self) -> HandleProgram:
        return HandleProgram(
            name="Voice Hub", description="Knows who is speaking, then passes the command to your agent",
            installed=True, version="1", supports_home_control=True,
            attribution=Attribution(name="Voice Hub", url=""),
            models=[HandleModel(name="voice_hub", description="Voice Hub", installed=True, version="1",
                                attribution=Attribution(name="Voice Hub", url=""), languages=[])])

    def _pick(self, device_id: str | None, person: str | None) -> str:
        return (self.settings["people"].get(person or "") or self.settings["devices"].get(device_id or "")
                or self.settings["default"] or DEFAULT_AGENT)

    async def handle(self, tr: Transcript):
        ctx = tr.context or {}
        device_id, satellite_id = ctx.get("device_id"), ctx.get("satellite_id")
        t0 = time.monotonic()
        person = None
        if self.capture is not None and device_id:
            person = await self.capture.speaker_now(device_id, WAIT_S)
        waited = (time.monotonic() - t0) * 1000
        agent_id = self._pick(device_id, person)

        # The real agent's conversation: the same as this run's, or the one a
        # reopened listen on this satellite continues.
        ha_conv = ctx.get("conversation_id")
        conv = self.threads.get(ha_conv or "")
        if conv is None and device_id and (fu := self.follow_up.pop(device_id, None)) and fu[0] > time.monotonic():
            conv = fu[1]
        text = tr.text or ""
        if agent_id == DEFAULT_AGENT and (fixed := canonical_timer(text)):
            _LOG.info("agent: timer request %r passed on as %r", text, fixed)
            text = fixed
        if self.settings["tell_name"] and agent_id != DEFAULT_AGENT and person and person != "unknown":
            text = f"({person} is speaking) {text}"
        key = (device_id, tr.text)
        if key in self._passing:                        # the chosen agent is Voice Hub itself
            _LOG.warning("agent: %s is Voice Hub itself; choose another agent on the page", agent_id)
            return NotHandled(text="Voice Hub cannot pass the command to itself: choose another agent.")
        msg = {"type": "conversation/process", "text": text, "agent_id": agent_id}
        for k, v in (("language", tr.language), ("device_id", device_id), ("satellite_id", satellite_id),
                     ("conversation_id", conv)):
            if v:
                msg[k] = v
        self._passing.add(key)
        try:
            _, res = await self.capture.ha.call(msg)
        except Exception as e:
            _LOG.warning("agent %s failed: %s", agent_id, e)
            return NotHandled(text=str(e))
        finally:
            self._passing.discard(key)
        resp = res.get("response") or {}
        speech = ((resp.get("speech") or {}).get("plain") or {}).get("speech", "")
        if ha_conv and res.get("conversation_id"):
            self.threads[ha_conv] = res["conversation_id"]
            if len(self.threads) > 200:
                self.threads.pop(next(iter(self.threads)))
        more = bool(res.get("continue_conversation"))
        sat = self.capture.satellite_of(device_id) if device_id else None
        if more and sat is not None:
            asyncio.create_task(self._reopen(device_id, sat, res.get("conversation_id")))
        self.last = ([{"time": time.time(), "device": sat.name if sat else device_id, "person": person,
                       "agent": agent_id, "waited_ms": round(waited), "text": tr.text, "reply": speech,
                       "follow_up": more}] + self.last)[:20]
        _LOG.info("agent: %s | %s -> %s (waited %.0f ms for the voiceprint)%s",
                  person or "?", agent_id, (speech or "")[:60], waited, " | follow-up" if more else "")
        if resp.get("response_type") == "error":
            return NotHandled(text=speech)
        return Handled(text=speech)

    async def _reopen(self, device_id: str, sat, conv: str | None) -> None:
        """Keep listening after a question: once the reply has played."""
        if not await self.capture.wait_idle(device_id, IDLE_TIMEOUT_S):
            return
        if conv:
            self.follow_up[device_id] = (time.monotonic() + FOLLOW_UP_S, conv)
        try:
            await self.capture.ha.call({"type": "call_service", "domain": "assist_satellite",
                                        "service": "start_conversation",
                                        "target": {"entity_id": sat.entity_id},
                                        "service_data": {"start_message": "", "preannounce": False}})
            _LOG.info("agent: %s listens again for the answer", sat.name)
        except Exception as e:
            self.follow_up.pop(device_id, None)
            _LOG.warning("could not reopen listening on %s: %s", sat.name, e)
