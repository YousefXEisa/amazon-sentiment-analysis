from pydantic import BaseModel, Field
from typing import Optional, Dict

class PredictRequest(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="Review title",
        json_schema_extra={
            "example": "Great Product!"
        }
    )
    content: str = Field(
        ...,
        description="Main body content of the review",
        json_schema_extra={
            "example": "I have been using this item for a month, works perfectly."
        }
    )
class PredictResponse(BaseModel):
    label: str = Field(
        ...,
        json_schema_extra={"example": "Positive"}
    )
    confidence: float = Field(
        ...,
        json_schema_extra={"example": 0.9854}
    )
    probabilities: Dict[str, float] = Field(
        ...,
        json_schema_extra={
            "example": {
                "Negative": 0.0146,
                "Positive": 0.9854
            }
        }
    )

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "Healthy"})
    model_repo: str = Field(..., json_schema_extra={"example": "YousefXEisa/amazon-roberta-sentiment"})
