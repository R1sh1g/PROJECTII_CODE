from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from io import BytesIO

from models.acd_infer import load_acd
from models.asc_infer import load_asc
from models.pipelines import ReviewPipeline
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="ReviewOps API")
from fastapi.middleware.cors import CORSMiddleware

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


acd = load_acd()
asc = load_asc()
pipeline = ReviewPipeline(acd, asc)

class ReviewInput(BaseModel):
    review: str
import csv
import pandas as pd
from io import BytesIO

def read_csv_safe(content: bytes) -> pd.DataFrame:
    # 1) decode bytes with fallback encodings
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            text = None
    if text is None:
        # last resort: replace bad chars
        text = content.decode("latin1", errors="replace")

    # 2) try normal read first (good CSVs)
    try:
        return pd.read_csv(
            BytesIO(text.encode("utf-8")),
            engine="python",
        )
    except Exception:
        pass

    # 3) try with flexible quoting + skipping bad lines
    # engine="python" allows on_bad_lines
    return pd.read_csv(
        BytesIO(text.encode("utf-8")),
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
    )


@app.post("/predict")
def predict_review(inp: ReviewInput):
    return pipeline.run(inp.review)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = read_csv_safe(content)

    if "review" not in df.columns:
        return {"error": "CSV must contain a 'review' column"}

    outputs = []
    for r in df["review"].astype(str):
        outputs.append(pipeline.run(r))

    return {
        "count": len(outputs),
        "results": outputs
    }


@app.get("/")
def health():
    return {"status": "ok"}
