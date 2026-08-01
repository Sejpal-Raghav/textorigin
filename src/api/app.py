from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.predictor import predict
from src.api.humanizer import humanize

app = FastAPI(
    title="TextOrigin",
    description="Classifies text as human-written, AI-written, or AI-written-then-paraphrased.",
    version="1.0.0",
)

# Enable CORS for Next.js frontend (default port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    human: float
    ai_written: float
    ai_paraphrased: float
    top_features: list[str]


class HumanizeRequest(BaseModel):
    text: str
    use_llm: bool = True
    similarity_threshold: float = 0.85


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(request.text) > 10_000:
        raise HTTPException(status_code=400, detail="Text too long (max 10,000 characters)")

    result = predict(request.text)
    return ClassifyResponse(**result)


@app.post("/humanize")
def api_humanize(request: HumanizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(request.text) > 10_000:
        raise HTTPException(status_code=400, detail="Text too long (max 10,000 characters)")

    return humanize(request.text, use_llm=request.use_llm, similarity_threshold=request.similarity_threshold)


@app.get("/health")
def health():
    return {"status": "ok"}
