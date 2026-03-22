from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="Predictive Maintenance API",
    description="ML-powered hard drive failure prediction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "F:/predictive-maintenance-digital-twin/models/xgboost_model.joblib"

model_data = None

def load_model():
    global model_data
    if model_data is None:
        model_data = joblib.load(MODEL_PATH)
    return model_data

class SMARTData(BaseModel):
    serial_number: str = Field(..., description="Drive serial number")
    model: Optional[str] = Field("unknown", description="Drive model")
    capacity_tb: Optional[float] = Field(0, description="Capacity in TB")
    age_days: Optional[int] = Field(0, description="Drive age in days")
    power_on_hours: Optional[int] = Field(0, description="Power on hours")
    
    smart_5_raw: Optional[float] = Field(0, description="Reallocated Sectors Count")
    smart_9_raw: Optional[float] = Field(0, description="Power-On Hours")
    smart_187_raw: Optional[float] = Field(0, description="Reported Uncorrectable Errors")
    smart_188_raw: Optional[float] = Field(0, description="Command Timeout")
    smart_194_raw: Optional[float] = Field(0, description="Temperature")
    smart_197_raw: Optional[float] = Field(0, description="Current Pending Sector Count")
    smart_198_raw: Optional[float] = Field(0, description="Offline Uncorrectable")
    smart_199_raw: Optional[float] = Field(0, description="UDMA CRC Error Count")
    
    smart_5_raw_mean_7d: Optional[float] = None
    smart_5_raw_std_7d: Optional[float] = None
    smart_5_raw_max_7d: Optional[float] = None
    smart_5_raw_delta_7d: Optional[float] = None
    smart_5_raw_mean_30d: Optional[float] = None
    
    smart_187_raw_mean_7d: Optional[float] = None
    smart_187_raw_std_7d: Optional[float] = None
    smart_187_raw_max_7d: Optional[float] = None
    smart_187_raw_delta_7d: Optional[float] = None
    smart_187_raw_mean_30d: Optional[float] = None
    
    smart_197_raw_mean_7d: Optional[float] = None
    smart_197_raw_std_7d: Optional[float] = None
    smart_197_raw_max_7d: Optional[float] = None
    smart_197_raw_delta_7d: Optional[float] = None
    smart_197_raw_mean_30d: Optional[float] = None
    
    smart_198_raw_mean_7d: Optional[float] = None
    smart_198_raw_std_7d: Optional[float] = None
    smart_198_raw_max_7d: Optional[float] = None
    smart_198_raw_delta_7d: Optional[float] = None
    
    temp_mean_7d: Optional[float] = None
    temp_max_7d: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "serial_number": "ZA10ABCD",
                "model": "ST12000NM0008",
                "capacity_tb": 12.0,
                "age_days": 365,
                "power_on_hours": 8760,
                "smart_5_raw": 10,
                "smart_9_raw": 8760,
                "smart_187_raw": 0,
                "smart_188_raw": 0,
                "smart_194_raw": 35,
                "smart_197_raw": 5,
                "smart_198_raw": 0,
                "smart_199_raw": 0
            }
        }


class PredictionResponse(BaseModel):
    serial_number: str
    failure_probability: float
    risk_level: str
    will_fail_30d: bool
    confidence: float
    recommendation: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    drives: List[SMARTData]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_type: str
    metrics: Dict[str, float]
    feature_count: int
    top_features: List[Dict[str, Any]]


def prepare_features(data: SMARTData) -> pd.DataFrame:
    
    features = {
        'smart_5_raw': data.smart_5_raw or 0,
        'smart_9_raw': data.smart_9_raw or data.power_on_hours or 0,
        'smart_187_raw': data.smart_187_raw or 0,
        'smart_188_raw': data.smart_188_raw or 0,
        'smart_194_raw': data.smart_194_raw or 0,
        'smart_197_raw': data.smart_197_raw or 0,
        'smart_198_raw': data.smart_198_raw or 0,
        'smart_199_raw': data.smart_199_raw or 0,
        
        'smart_5_raw_mean_7d': data.smart_5_raw_mean_7d if data.smart_5_raw_mean_7d is not None else data.smart_5_raw or 0,
        'smart_5_raw_std_7d': data.smart_5_raw_std_7d or 0,
        'smart_5_raw_max_7d': data.smart_5_raw_max_7d if data.smart_5_raw_max_7d is not None else data.smart_5_raw or 0,
        'smart_5_raw_delta_7d': data.smart_5_raw_delta_7d or 0,
        
        'smart_187_raw_mean_7d': data.smart_187_raw_mean_7d if data.smart_187_raw_mean_7d is not None else data.smart_187_raw or 0,
        'smart_187_raw_std_7d': data.smart_187_raw_std_7d or 0,
        'smart_187_raw_max_7d': data.smart_187_raw_max_7d if data.smart_187_raw_max_7d is not None else data.smart_187_raw or 0,
        'smart_187_raw_delta_7d': data.smart_187_raw_delta_7d or 0,
        
        'smart_197_raw_mean_7d': data.smart_197_raw_mean_7d if data.smart_197_raw_mean_7d is not None else data.smart_197_raw or 0,
        'smart_197_raw_std_7d': data.smart_197_raw_std_7d or 0,
        'smart_197_raw_max_7d': data.smart_197_raw_max_7d if data.smart_197_raw_max_7d is not None else data.smart_197_raw or 0,
        'smart_197_raw_delta_7d': data.smart_197_raw_delta_7d or 0,
        
        'smart_198_raw_mean_7d': data.smart_198_raw_mean_7d if data.smart_198_raw_mean_7d is not None else data.smart_198_raw or 0,
        'smart_198_raw_std_7d': data.smart_198_raw_std_7d or 0,
        'smart_198_raw_max_7d': data.smart_198_raw_max_7d if data.smart_198_raw_max_7d is not None else data.smart_198_raw or 0,
        'smart_198_raw_delta_7d': data.smart_198_raw_delta_7d or 0,
        
        'smart_5_raw_mean_30d': data.smart_5_raw_mean_30d if data.smart_5_raw_mean_30d is not None else data.smart_5_raw or 0,
        'smart_187_raw_mean_30d': data.smart_187_raw_mean_30d if data.smart_187_raw_mean_30d is not None else data.smart_187_raw or 0,
        'smart_197_raw_mean_30d': data.smart_197_raw_mean_30d if data.smart_197_raw_mean_30d is not None else data.smart_197_raw or 0,
        
        'temp_mean_7d': data.temp_mean_7d if data.temp_mean_7d is not None else data.smart_194_raw or 0,
        'temp_max_7d': data.temp_max_7d if data.temp_max_7d is not None else data.smart_194_raw or 0,
        
        'power_on_hours': data.power_on_hours or data.smart_9_raw or 0,
        'age_days': data.age_days or 0,
        'capacity_tb': data.capacity_tb or 0,
    }
    
    return pd.DataFrame([features])


def get_risk_level(probability: float) -> tuple:
    if probability >= 0.7:
        return "CRITICAL", "Immediate replacement recommended"
    elif probability >= 0.5:
        return "HIGH", "Schedule replacement within 7 days"
    elif probability >= 0.3:
        return "MEDIUM", "Monitor closely, plan replacement within 30 days"
    elif probability >= 0.1:
        return "LOW", "Continue monitoring, routine maintenance"
    else:
        return "HEALTHY", "No action required"


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Predictive Maintenance API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        load_model()
        model_loaded = True
    except:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    try:
        data = load_model()
        
        importance = data['model'].feature_importances_
        feature_names = data['feature_names']
        
        top_features = sorted(
            [{"feature": f, "importance": float(i)} for f, i in zip(feature_names, importance)],
            key=lambda x: x['importance'],
            reverse=True
        )[:10]
        
        return ModelInfoResponse(
            model_type=data['model_type'],
            metrics={k: float(v) for k, v in data['metrics'].items()},
            feature_count=len(feature_names),
            top_features=top_features
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: SMARTData):
    try:
        model_data = load_model()
        model = model_data['model']
        scaler = model_data['scaler']
        
        features = prepare_features(data)
        
        feature_names = model_data['feature_names']
        features = features.reindex(columns=feature_names, fill_value=0)
        
        features_scaled = scaler.transform(features)
        probability = float(model.predict_proba(features_scaled)[0, 1])
        
        risk_level, recommendation = get_risk_level(probability)
        
        return PredictionResponse(
            serial_number=data.serial_number,
            failure_probability=round(probability, 4),
            risk_level=risk_level,
            will_fail_30d=probability >= 0.5,
            confidence=round(abs(probability - 0.5) * 2, 4),
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    predictions = []
    
    for drive in request.drives:
        pred = await predict(drive)
        predictions.append(pred)
    
    probabilities = [p.failure_probability for p in predictions]
    risk_counts = {}
    for p in predictions:
        risk_counts[p.risk_level] = risk_counts.get(p.risk_level, 0) + 1
    
    summary = {
        "total_drives": len(predictions),
        "avg_failure_probability": round(np.mean(probabilities), 4),
        "max_failure_probability": round(max(probabilities), 4),
        "drives_at_risk": sum(1 for p in predictions if p.failure_probability >= 0.3),
        "risk_distribution": risk_counts
    }
    
    return BatchPredictionResponse(predictions=predictions, summary=summary)


@app.on_event("startup")
async def startup_event():
    print("Loading ML model...")
    load_model()
    print("Model loaded successfully!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
