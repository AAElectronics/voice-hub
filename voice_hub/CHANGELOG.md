# Changelog

## 0.11.0

- The repository moved to the AAElectronics organization: in Home Assistant, **Settings → Add-ons → Add-on Store → ⋮ → Repositories**, replace `https://github.com/ale1800/voice-hub` with `https://github.com/AAElectronics/voice-hub` (the old address keeps working through GitHub's redirect for now).
- "Who spoke?" menu on every saved command: picking a name adds that command to the person's voice samples (or moves it from another person); "New person…" starts a new one.
- LocalDot boards that recognise speakers themselves (development firmware with the ESP32-P4 voiceprint): the board's own guess is shown next to Voice Hub's, its voiceprint is kept with the command, and whenever people change the board gets its people list again (actions `p4_people_begin` / `p4_people_chunk` / `p4_people_finish`). Board voiceprints and Voice Hub voiceprints are never mixed.

## 0.10.2

- Agent: Italian timer requests that Home Assistant's built-in sentences miss are passed on in a form it recognises ("imposto/metti/fammi un timer di 5 minuti", "imposta il timer a 5 minuti", "5 minuti di timer", durations in words such as "un'ora", "mezz'ora", "2 minuti e mezzo"). Only when the agent is Home Assistant's own and a duration follows; other commands are untouched.

## 0.10.1

- The interface and the enrollment phrases follow Home Assistant's language when the `language` option is empty (Italian or English; other languages use English).

## 0.8.3

- the Wake word tab shows the right time for each check (it showed a 1970 date).

## 0.8.2

- the log no longer calls the name "too late" when Voice Hub is the agent (it waited for it before passing the command on).

## 0.8.1

- second-stage wake word computes openWakeWord features as the audio streams in (one embedding per 80 ms, as openWakeWord itself; scores identical to it): 17 ms per step instead of 270, so the check keeps up with the audio and the command after the wake word reaches speech-to-text. The old whole-window computation also under-scored some real wakes.

## 0.8.0

- speaker model NVIDIA TitaNet large (CC-BY-4.0) instead of WeSpeaker ResNet34 (equal error rate 37% -> 1.9% on our enrolled samples); enrolled samples recomputed from their recordings; no cohort subtraction (it removed the main speaker's own voice); same-satellite samples first; threshold 0.35, margin 0.10.

## 0.7.0

- Voice Hub as a conversation agent (Wyoming handle on 10400): waits for the voiceprint, publishes the speaker, then passes the command to any agent (per satellite / per person, name told to LLM agents); reopens listening for LLM follow-up questions. Agent tab.

## 0.6.3

- a Wake word tab for the second stage; satellite cards fold (closed: name, online, Call, state).

## 0.6.2

- second-stage wake word: upload your own openWakeWord .onnx models from the page, switch between them live (Home Assistant keeps the same wake word name), remove uploaded ones.

## 0.6.1

- a Call button next to each LocalDot with an intercom (VoIP Stack browser phone, destination preselected), online/offline and in-call status; labels in Home Assistant's language.

## 0.5.1

- voiceprint computed while the person speaks (every 0.5 s of audio), 3 threads.

## 0.5.0

- satellite signature subtracted (cohort mean), top-3 samples per person; duplicate samples removed;
  one Save at a time.

## 0.4.1

- a step counts only when the transcript matches the phrase (green, next).

## 0.4.0

- guided enrollment (12 phrases in order, conditions, redo/skip, review);
  satellite statistics grouped, sensors only.

## 0.3.0

- sidebar page (satellites, people with "Add person", recent commands).

## 0.2.0

- pipeline capture mode, the default. Running on a Home Assistant
  Yellow on two satellites; how often it beats the intent stage is still to be
  measured on real commands.

