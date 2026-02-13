# reviewOps/backend/app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import csv
import pandas as pd
from io import BytesIO

from models.acd_infer import load_acd
from models.asc_infer import load_asc
from models.pipelines import ReviewPipeline


app = FastAPI(title="ReviewOps API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewInput(BaseModel):
    review: str


def read_csv_safe(content: bytes) -> pd.DataFrame:
    # 1) decode bytes with fallback encodings
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    if text is None:
        text = content.decode("latin1", errors="replace")

    # 2) try normal read first
    try:
        return pd.read_csv(BytesIO(text.encode("utf-8")), engine="python")
    except Exception:
        pass

    # 3) fallback with flexible quoting + skipping bad lines
    return pd.read_csv(
        BytesIO(text.encode("utf-8")),
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
    )


@app.on_event("startup")
def startup_load_models():

    acd = load_acd()
    asc = load_asc()
    app.state.pipeline = ReviewPipeline(acd, asc)


@app.post("/predict")
def predict_review(inp: ReviewInput):
    return app.state.pipeline.run(inp.review)


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = read_csv_safe(content)

    if "review" not in df.columns:
        return {"error": "CSV must contain a 'review' column"}

    outputs = [app.state.pipeline.run(str(r)) for r in df["review"].astype(str)]

    return {"count": len(outputs), "results": outputs}


@app.get("/")
def health():
    return {"status": "ok"}
