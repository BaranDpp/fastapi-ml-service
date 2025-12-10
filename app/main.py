from fastapi import FastAPI
from app.api.v1.router import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Config-Driven ML API"
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "System Online. Go to /docs"}