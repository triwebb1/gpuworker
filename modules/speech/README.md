# speech

Text-to-speech in the voice of a reference recording, and a quality-check job type a client
can submit for each result. Two job types share one venv (`/opt/venvs/speech`):

| Job type | Mode | Sandbox | What it does |
|---|---|---|---|
| `speech.voxcpm2` | serve | `strict` | Speaks text with [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) in the voice of a short reference recording, the persona. Jobs that pass the same recording get the same voice. Long text is split into chunks of whole sentences, and the chunks are joined into one WAV. The model loads once per warm container. |
| `speech.check` | oneshot | `standard` | The quality check. Transcribes a WAV with faster-whisper large-v3 and scores it against the text it should contain: word error rate, and how many words are missing from the end. The record says whether it passed your thresholds. |

The check is a separate job type so that loading Whisper never inflates the footprint
the broker learns for the text-to-speech type; the reasoning is in
[`src/speech/check.py`](src/speech/check.py). The options for each type are documented
at the top of [`src/speech/tts.py`](src/speech/tts.py) and
[`src/speech/check.py`](src/speech/check.py).

## Staging the weights

Containers run with no network, so each type's files are staged on the broker host,
sealed, and mounted read-only at `/job/weights` (MODULE_GUIDE 3.3). With the
`huggingface_hub` CLI (`hf`), from the repository root:

```bash
hf download openbmb/VoxCPM2 --local-dir weights/speech.voxcpm2/VoxCPM2
hf download Systran/faster-whisper-large-v3 --local-dir weights/speech.check/faster-whisper-large-v3
gpuworker stage-weights speech.voxcpm2
gpuworker stage-weights speech.check
```

`speech.voxcpm2` is `strict`, so it also needs a pinned image (`images/build.sh --pin`).
The worker turns off VoxCPM's optional denoiser, which is a second model that is not
staged. It does not use VoxCPM's text normalization, which fetches data at run time.

## A job

```json
{
  "job_id": "welcome-01",
  "job_type": "speech.voxcpm2",
  "inputs": ["in/persona.wav"],
  "outputs": ["out/welcome-01.wav", "out/welcome-01.json"],
  "options": {
    "text": "Welcome back. Here is what changed since your last visit.",
    "reference": "in/persona.wav",
    "reference_text": "What the persona recording says, word for word.",
    "seed": 7
  }
}
```

The persona recording is a few seconds of clean speech in the voice you want. With its
transcript in `reference_text`, the model also continues from the recording, which holds
the voice closer to it. Without `reference`, the model chooses a voice.

To check the result, submit a `speech.check` job with the WAV as an input and
`"options": {"audio": "<path>", "text": "<the same text>"}`. The check exits 0 whenever it
writes its record, including a record with `"passed": false`, so a client reads the
verdict from the record and can resubmit the line.

A job is bounded by `serve_job_timeout` (1800 s). Split a long text across several jobs;
the voice stays the same as long as each job passes the same persona recording.

## Smoke tests

- `smoke/speech.voxcpm2.sh` renders one sentence in a voice the model chooses.
- `smoke/speech.voxcpm2-persona.sh <wav> "<transcript>"` renders another sentence in the
  voice of `<wav>`; the WAV from the first test works as a persona.
- `smoke/speech.check.sh <wav> "<text>"` checks a rendered WAV against its text.

## Footprint

The `vram_mb` values in `module.toml` are the declared footprints. The broker measures
each type's peak when it loads alone and raises the footprint if the declaration was low.
