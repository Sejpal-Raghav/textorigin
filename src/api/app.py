from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.predictor import predict

app = FastAPI(
    title="TextOrigin",
    description="Classifies text as human-written, AI-written, or AI-written-then-paraphrased.",
    version="1.0.0",
)


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    human: float
    ai_written: float
    ai_paraphrased: float
    top_features: list[str]


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(request.text) > 10_000:
        raise HTTPException(status_code=400, detail="Text too long (max 10,000 characters)")

    result = predict(request.text)
    return ClassifyResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}
