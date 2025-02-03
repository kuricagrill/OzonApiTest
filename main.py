import requests
from transformers import pipeline
from fastapi import FastAPI

# 🔹 Ozon API - Подключение
OZON_API_URL = "https://api-seller.ozon.ru"
HEADERS = {
    "Client-Id": "29201731",
    "Api-Key": "d3a93409-a7a2-4a8f-ae1e-7dfdd19d894b",
    "Content-Type": "application/json"
}

# 🔹 Функция поиска товаров конкурентов
def search_products(keyword, limit=50):
    url = f"{OZON_API_URL}/v1/product/list"
    data = {"search": keyword, "limit": limit}
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

# 🔹 Фильтр - выбираем лучшие товары
def find_best_products(products):
    best = [p for p in products if p.get('sales', 0) > 1000 and p.get('rating', 0) > 4.5]
    return best

# 🔹 Подключение нейросети Hugging Face
ai_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

# 🔹 Функция анализа товаров нейросетью
def analyze_products(products):
    context = "\n".join([
        f"{p['name']}, цена: {p['price']} руб., продаж: {p.get('sales', 0)}" 
        for p in products
    ])
    question = "Какие товары лучше всего продаются?"
    answer = ai_model(question=question, context=context)
    return answer

# 🔹 FastAPI сервер
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Ozon AI Analyzer работает!"}

@app.get("/search")
def search(keyword: str):
    products = search_products(keyword)
    return {"products": products.get("result", [])}

@app.get("/best-products")
def best_products(keyword: str):
    products = search_products(keyword).get("result", [])
    best = find_best_products(products)
    return {"best_products": best}

@app.get("/analyze")
def analyze(keyword: str):
    products = search_products(keyword).get("result", [])
    answer = analyze_products(products)
    return {"analysis": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
