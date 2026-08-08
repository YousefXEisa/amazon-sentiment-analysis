from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from typing import Optional

from src.inference import SentimentPredictor, HF_REPO_ID
from api.schemas import PredictRequest, PredictResponse, HealthResponse

predictor : Optional[SentimentPredictor]

@asynccontextmanager
async def lifespan(app:FastAPI):
    global predictor
    print(f"Initializing API: Loading model from Hugging Face ({HF_REPO_ID})...")
    try:
        predictor = SentimentPredictor(model_path=HF_REPO_ID)
        print("Model loaded successfully into memory. API is ready for inference!")
    except Exception as e:
        print(f"Failed to load model on startup: {str(e)}")
        predictor= None

    yield
    print("Shutting down Sentiment API Service...")
    predictor= None

app = FastAPI(
    title="Amazon Reviews Sentiment Analysis API",
    description=(
        "Production-ready FastAPI service for Amazon Product Review Sentiment Analysis. "
        "Powered by a Fine-Tuned RoBERTa model hosted on Hugging Face Hub."
    ),
    version="1.0.5",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

@app.get("/",
         summary="Root Endpoint",
         tags=["General"]
         )
def root():
    return {
        "message": "Welcome to the Amazon Reviews Sentiment Analysis API!",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health",
         response_model=HealthResponse,
         summary="Health Check",
         tags=["General"]
         )
def health_check():
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail = "Model service is currently unavailable or failed to initialize.")
    return {
        "status": "healthy",
        "model_repo": HF_REPO_ID
    }

@app.post("/predict",
         response_model=PredictResponse,
         summary="Predict Sentiment",
         tags=["Inference"]
         )
def predict_sentiment(payload: PredictRequest):

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not loaded."
        )

    title_str = payload.title.strip() if payload.title is not None else ""
    content_str = payload.content.strip() if payload.content is not None else ""

    if not title_str and not content_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'title' or 'content' must contain non-empty text."
        )

    try:
        result = predictor.predict(content=payload.content, title=payload.title)
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail= str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference execution: {str(e)}"
        )

