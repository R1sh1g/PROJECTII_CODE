# reviewOps/backend/models/acd_infer.py

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _resolve_path(maybe_path: str | None, base_dir: Path) -> Path | None:

    if not maybe_path:
        return None
    p = Path(maybe_path)
    return p if p.is_absolute() else (base_dir / p)


def _backend_root() -> Path:

    return Path(__file__).resolve().parents[1]


def _load_label_names(label_path: Path) -> list[str]:

    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict):
        labels = data.get("label_names")
        if not isinstance(labels, list):
            raise ValueError(
                f"label_names.json dict must contain a list under key 'label_names'. "
                f"Got type={type(labels)} at {label_path}"
            )
    else:
        raise ValueError(
            f"label_names.json must be a list or a dict with 'label_names'. "
            f"Got type={type(data)} at {label_path}"
        )

    # Normalize to list[str]
    labels = [str(x).strip() for x in labels if str(x).strip()]
    if not labels:
        raise ValueError(f"label_names.json at {label_path} contains no valid labels.")
    return labels


class ACDInfer:
    def __init__(self, model_dir: str | Path, threshold: float = 0.5, max_len: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)
        self.max_len = int(max_len)

        model_dir = Path(model_dir).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"ACD model_dir not found: {model_dir}")

        # Load true label order saved during training
        label_path = model_dir / "label_names.json"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label_names.json in {model_dir}. "
                "Ensure it's inside the ACD model folder."
            )

        self.label_names = _load_label_names(label_path)

        # HuggingFace expects str paths
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

        # Safety: ensure label count matches model outputs
        n_model = int(self.model.config.num_labels)
        if len(self.label_names) != n_model:
            raise ValueError(
                f"Label mismatch: label_names.json has {len(self.label_names)} labels "
                f"but model expects {n_model}. Check saved artifacts."
            )

    @torch.no_grad()
    def predict(self, review_text: str):
        enc = self.tokenizer(
            review_text,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        logits = out.logits.squeeze(0)  # [num_labels]
        probs = torch.sigmoid(logits)   # multi-label probs
        probs_list = probs.tolist()

        scored = [(self.label_names[i], float(p)) for i, p in enumerate(probs_list)]
        scored.sort(key=lambda x: x[1], reverse=True)

        # --- Controls ---
        TOP_K = 3
        MIN_PROB = 0.30
        THRESH = self.threshold
        MISC_LABEL = "miscellaneous"
        MISC_MIN = 0.80

        picked = [x for x in scored if x[1] >= THRESH]

        if len(picked) < 1:
            picked = [x for x in scored if x[1] >= MIN_PROB][:TOP_K]

        picked = [x for x in picked if not (x[0] == MISC_LABEL and x[1] < MISC_MIN)]

        if not picked:
            non_misc = [x for x in scored if x[0] != MISC_LABEL and x[1] >= MIN_PROB]
            picked = non_misc[:TOP_K] if non_misc else scored[:1]

        return picked[:TOP_K]


def load_acd():

    backend_root = _backend_root()
    models_dir = backend_root / "models"
    artifacts_dir = models_dir / "artifacts"

    env_model_dir = _resolve_path(os.getenv("ACD_MODEL_DIR"), backend_root)
    default_model_dir = artifacts_dir / "acd_model"

    model_dir = env_model_dir or default_model_dir

    return ACDInfer(model_dir=model_dir, threshold=0.5, max_len=256)
