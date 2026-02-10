# reviewOps/backend/models/acd_infer.py

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ACDInfer:
    def __init__(self, model_dir: str, threshold: float = 0.5, max_len: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.max_len = max_len

        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"ACD model_dir not found: {model_dir}")

        # ✅ Load true label order saved during training
        label_path = os.path.join(model_dir, "label_names.json")
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Missing label_names.json in {model_dir}. "
                "Your training script saves it — ensure it's inside acd_model folder."
            )

        with open(label_path, "r", encoding="utf-8") as f:
            self.label_names = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(self.device)
        self.model.eval()

        # Safety: ensure label count matches model outputs
        n_model = int(self.model.config.num_labels)
        if len(self.label_names) != n_model:
            raise ValueError(
                f"Label mismatch: label_names.json has {len(self.label_names)} labels "
                f"but model expects {n_model}. Check your saved artifacts."
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
        logits = out.logits.squeeze(0)          # [num_labels]
        probs = torch.sigmoid(logits)           # multi-label probs
        probs_list = probs.tolist()
        scored = [(self.label_names[i], float(p)) for i, p in enumerate(probs_list)]
        scored.sort(key=lambda x: x[1], reverse=True)
        # --- Controls (tune for demo) ---
        TOP_K = 3
        MIN_PROB = 0.30          # allow weaker multi-label picks
        THRESH = self.threshold  # keep 0.50 if you want, but MIN_PROB makes it stable
        MISC_LABEL = "miscellaneous"
        MISC_MIN = 0.80          # only show misc if very confident

        picked = [x for x in scored if x[1] >= THRESH]

        # If threshold yields too few labels, backfill from top scores
        if len(picked) < 1:
            picked = [x for x in scored if x[1] >= MIN_PROB][:TOP_K]

        # Suppress misc unless very high confidence
        picked = [x for x in picked if not (x[0] == MISC_LABEL and x[1] < MISC_MIN)]

        # If everything got removed (misc only), show best non-misc if available
        if not picked:
            non_misc = [x for x in scored if x[0] != MISC_LABEL and x[1] >= MIN_PROB]
            if non_misc:
                picked = non_misc[:TOP_K]
            else:
                # last resort: show top-1 (even if misc)
                picked = scored[:1]

        # Cap to TOP_K
        picked = picked[:TOP_K]
        return picked



def load_acd():
    here = os.path.dirname(__file__)  # .../reviewOps/backend/models
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))  # .../PROJECTII_CODE
    model_dir = os.path.join(project_root, "models", "ACD", "outputs", "acd_model")
    return ACDInfer(model_dir=model_dir, threshold=0.5, max_len=256)
