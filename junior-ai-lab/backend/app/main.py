from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.database import Base, engine, SessionLocal, migrate_schema
from app.routers import auth, dashboard, submissions, synthesize, management, courses, recognition
from app.routers.ai_evaluation import router as ai_evaluation_router
from app.routers.analytics import router as analytics_router
from app.routers.cnn import race, leaderboard, dataset, model_design, model_test, statistics, evaluation, materials
from app.seed import seed_database

RESET_FLAG = os.path.join(os.path.dirname(__file__), ".reset_db")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.path.exists(RESET_FLAG):
        # Full reset requested — drop and re-seed
        Base.metadata.drop_all(bind=engine)
        os.remove(RESET_FLAG)
    # Always ensure tables exist; seed only if empty
    Base.metadata.create_all(bind=engine)
    migrate_schema()
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()
    yield


is_production = os.getenv("APP_ENV", "development").lower() == "production"
app = FastAPI(
    title="AI Platform API",
    version="2.1.0",
    lifespan=lifespan,
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
    openapi_url=None if is_production else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(submissions.router)
app.include_router(dashboard.router)
app.include_router(synthesize.router)
app.include_router(management.router)
app.include_router(ai_evaluation_router)
app.include_router(analytics_router)
app.include_router(courses.router)
app.include_router(recognition.router)

app.include_router(race.router)
app.include_router(leaderboard.router)
app.include_router(dataset.router)
app.include_router(model_design.router)
app.include_router(model_test.router)
app.include_router(statistics.router)
app.include_router(evaluation.router)
app.include_router(materials.router)  # 添加资料下载路由
app.include_router(evaluation.router)
app.include_router(materials.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
