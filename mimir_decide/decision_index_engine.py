"""Engine for the Decision Index kit (https://github.com/apolinario/decision-index, MIT): our models as an `Engine`.

    pip install -e <decision-index checkout>            # in the same environment as mimir_decide
    python -m decision_index run --engine mimir_decide.decision_index_engine:MimirDecideEngine \\
        --option run_dir=runs/slot-v0 --option checkpoint=best --rows ... --out runs/di-slot-v0

Options (all strings, as `--option key=value`): run_dir (required), checkpoint (default best; `zero-shot` for the
untrained letter model), max_len (default: the value stored in the checkpoint), calibrated (default true: the per-kind
temperature of calibration.json), batch_size (default 16), device (default: cuda, mps or cpu).

Mapping (D29): a `choice` question is a Choice decision whose options are the criteria descriptions in the given
order; a `noul` question is a Noul decision with options ["yes", "no"] and the answer is P(yes). The state is used as
is (JSON objects are serialised compactly), the instructions are the question. The kit's hard rules hold here: no
truncation (a request that does not fit raises `Unsupported`), no option pruning (more than 26 options raise
`Unsupported` for the letter model), no prompt change per benchmark. The test split is not involved.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

try:  # the kit is optional: tests and the rest of the package work without it
    from decision_index.engines import Engine, Unsupported
except ImportError:  # pragma: no cover - replaced by the kit's classes when it is installed
    class Unsupported(ValueError):
        pass

    class Engine:
        name = "engine"
        provenance: dict = {}
        latency = ""

        def __init__(self, **options):
            self.options = options

        def warmup(self):
            pass

        def runtime(self):
            return {}

        def synchronize(self):
            pass

        def close(self):
            pass

from .formatting import LETTERS, MAX_OPTION_TOKENS, Formatter, mode_for
from .inference import device_auto, predict_logits
from .schema import Decision


def _text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def _flag(v, default: bool) -> bool:
    return default if v is None else str(v).lower() in ("1", "true", "yes")


def to_decisions(state, questions: dict) -> tuple[list[str], list[Decision], list[list[str]]]:
    """Kit request -> (question keys, decisions, option keys per question). Targets are placeholders (never used)."""
    keys, decs, opt_keys = [], [], []
    st = _text(state)
    for k, q in questions.items():
        if q["type"] == "choice":
            ok = list(q["criteria"])
            options = [_text(q["criteria"][o]) for o in ok]
            kind = "choice"
        elif q["type"] == "noul":
            ok, options, kind = ["yes", "no"], ["yes", "no"], "noul"
        else:
            raise Unsupported("Unsupported question type " + str(q["type"]))
        n = len(options)
        decs.append(Decision(id=k, case_id=k, group_id=k, dataset="decision-index", source="decision-index/request",
                             split="test", language="und", license="n/a", kind=kind, state=st,
                             question=_text(q.get("instructions", "")), options=options, target=[1.0 / n] * n))
        keys.append(k)
        opt_keys.append(ok)
    return keys, decs, opt_keys


def softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits.astype(np.float64) / T
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


class MimirDecideEngine(Engine):
    name = "mimir-decide"
    latency = "In-process request wall time including prompt construction; excludes model loading. Questions of one request run as one batch."

    def __init__(self, run_dir=None, checkpoint="best", max_len=None, calibrated=None, batch_size=16, device=None,
                 **options):
        super().__init__(**options)
        from .model import load_model

        if run_dir is None:
            raise ValueError("--option run_dir=runs/<run> is required")
        self.run = Path(run_dir)
        self.device = device or device_auto()
        self.model, self.tok, self.meta = load_model(self.run / checkpoint, self.device)
        self.max_len = int(max_len) if max_len else int(self.meta["max_len"])
        self.batch_size = int(batch_size)
        self.mode = mode_for(self.meta["model_type"])
        cal = self.run / "calibration.json"
        self.temps = json.loads(cal.read_text())["temperatures"] if _flag(calibrated, True) and cal.exists() else None
        # a formatter without a length limit: lets us measure the full prompt and refuse instead of truncating
        self.fmt = Formatter(self.tok, self.mode, 10 ** 9, self.meta["marker_id"])
        self.provenance = {"repo": "mimir-decide", "run_dir": str(self.run), "checkpoint": checkpoint,
                           "model_type": self.meta["model_type"], "max_len": self.max_len,
                           "calibrated": self.temps is not None, "temperatures": self.temps,
                           "base_model": self.meta.get("base_model"), "revision": self.meta.get("revision")}

    def _check_fits(self, d: Decision) -> None:
        k = len(d.options)
        if self.mode == "letter" and k > len(LETTERS):
            raise Unsupported(f"letter model supports at most {len(LETTERS)} options, got {k}")
        for j, o in enumerate(d.options):
            text = f" {o}\n" if self.mode == "slot" else o if self.mode == "scalar" else f"{LETTERS[j]}. {o}\n"
            if len(self.fmt._ids(text)) > MAX_OPTION_TOKENS:
                raise Unsupported(f"option {j} has more than {MAX_OPTION_TOKENS} tokens (the formatter would cut it)")
        enc = self.fmt.encode(d)
        if enc is None or len(enc.input_ids) > self.max_len:
            raise Unsupported(f"prompt of {len(enc.input_ids) if enc else '>'} tokens exceeds max_len {self.max_len}")

    def __call__(self, state, questions):
        keys, decs, opt_keys = to_decisions(state, questions)
        for d in decs:
            self._check_fits(d)
        logits, kept = predict_logits(self.model, self.tok, self.meta, decs, self.batch_size, self.device,
                                      autocast=self.device.startswith("cuda"), max_len=self.max_len)
        if len(kept) != len(decs):
            raise Unsupported("a question did not fit into the context window")
        answers = {}
        for key, d, ok, lg in zip(keys, decs, opt_keys, logits):
            p = softmax_T(lg[: len(d.options)], (self.temps or {}).get(d.kind, 1.0))
            if not np.all(np.isfinite(p)) or not math.isclose(float(p.sum()), 1.0, abs_tol=1e-6):
                raise ValueError("non-finite probabilities")
            if d.kind == "noul":
                answers[key] = {"type": "noul", "noul": float(p[0])}   # option 0 is "yes"
            else:
                probs = {o: float(x) for o, x in zip(ok, p)}
                answers[key] = {"type": "choice", "choice": max(probs, key=lambda o: probs[o]), "probabilities": probs}
        return {"model": f"mimir-decide:{self.run.name}", "answers": answers}, None
