import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Тест проверки работоспособности сервиса"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success():
    """Happy Path: тест успешного предсказания"""
    test_data = {
        "age": 35,
        "income": 65000.0,
        "months_on_book": 24,
        "credit_limit": 15000.0
    }
    
    response = client.post("/predict", json=test_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Проверяем структуру ответа
    assert "prediction" in data
    assert "score" in data
    assert "model_version" in data
    
    # Проверяем допустимые значения
    assert data["prediction"] in ["low_risk", "high_risk"]
    assert 0 <= data["score"] <= 1
    assert data["model_version"] == "1.0-tabular"


def test_predict_bad_input():
    """Bad Input: тест обработки неверных данных"""
    # Неполные данные
    bad_data = {
        "age": 35,
        "income": 65000.0
        # Отсутствуют обязательные поля
    }
    
    response = client.post("/predict", json=bad_data)
    assert response.status_code == 422  # Validation Error


def test_predict_invalid_age():
    """Тест невалидного возраста"""
    test_data = {
        "age": 15,  # Возраст меньше минимального
        "income": 65000.0,
        "months_on_book": 24,
        "credit_limit": 15000.0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422


def test_predict_negative_income():
    """Тест отрицательного дохода"""
    test_data = {
        "age": 35,
        "income": -1000.0,  # Отрицательный доход
        "months_on_book": 24,
        "credit_limit": 15000.0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422


def test_root_endpoint():
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
