import random
from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the client (18-100)")
    income: float = Field(..., ge=0, description="Annual income of the client")
    months_on_book: int = Field(..., ge=0, description="Number of months as bank client")
    credit_limit: float = Field(..., ge=0, description="Credit limit of the client")


class PredictionResponse(BaseModel):
    prediction: str
    score: float
    model_version: str


def load_model():
    """Симуляция загрузки модели для табличных данных."""
    print("Загрузка скоринговой модели...")
    return {"version": "1.0-tabular"}


def predict(model: dict, features: ClientFeatures) -> dict:
    """Симуляция предсказания на основе данных клиента."""
    # Более сложная логика для реалистичности
    base_score = 0.5
    
    # Влияние возраста
    if features.age < 25:
        base_score -= 0.1
    elif features.age > 45:
        base_score += 0.1
        
    # Влияние дохода
    income_factor = min(features.income / 100000, 0.3)
    base_score += income_factor
    
    # Влияние длительности отношений с банком
    months_factor = min(features.months_on_book / 60, 0.2)
    base_score += months_factor
    
    # Влияние кредитного лимита
    limit_factor = min(features.credit_limit / 50000, 0.2)
    base_score += limit_factor
    
    # Нормализация и добавление случайности
    score = max(0.1, min(0.99, base_score + random.uniform(-0.1, 0.1)))
    
    if score > 0.7:
        prediction = "low_risk"
    else:
        prediction = "high_risk"
        
    return {
        "prediction": prediction, 
        "score": round(score, 4),
        "model_version": model["version"]
    }
