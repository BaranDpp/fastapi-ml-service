from fastapi import FastAPI
from app.api.v1.router import api_router  # <-- Düzeltilen satır (Nokta yok)
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="TensorFlow Keras Model Training & Inference API"
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "ML API is running!"}