from fastapi import FastAPI, HTTPException
from app.model import ClientFeatures, PredictionResponse, load_model, predict

app = FastAPI(
    title="Credit Scoring API",
    description="API для предсказания кредитного риска клиентов",
    version="1.0.0"
)

# Глобальная переменная для хранения модели
model = None


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    global model
    model = load_model()


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Добро пожаловать в Credit Scoring API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(features: ClientFeatures):
    """
    Предсказание кредитного риска клиента
    
    - **age**: Возраст клиента (18-100)
    - **income**: Годовой доход
    - **months_on_book**: Количество месяцев как клиент банка  
    - **credit_limit**: Кредитный лимит
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = predict(model, features)
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
