import requests


API_URL = "http://127.0.0.1:8000"


def test_health():
    response = requests.get(f"{API_URL}/health")

    print("Health:", response.json())

    assert response.status_code == 200


def test_prediction():
    payload = {
        "title": "Excellent product",
        "content": "I really love this product. The quality is amazing!"
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload
    )

    print("Prediction:", response.json())

    assert response.status_code == 200


if __name__ == "__main__":
    test_health()
    test_prediction()