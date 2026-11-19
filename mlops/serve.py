import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn

# Request payload validation schema
class PredictRequest(BaseModel):
    claimed_skill: float = Field(..., ge=0.0, le=10.0, description="Resume claimed rating (0.0 to 10.0)")
    measured_ability: float = Field(..., ge=0.0, le=10.0, description="Actual measured ability (0.0 to 10.0)")

# Response payload schema
class PredictResponse(BaseModel):
    inflated: bool
    difference: float

def load_champion_model():
    """
    MLflow Model Registry se latest 'champion' model load karta hai.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:6001")

    # MLflow server ka URI set kar rahe hain jahan champion model registered hai
    mlflow.set_tracking_uri(tracking_uri)

    # MLflow model registry se 'champion' alias wala model load kar rahe hain
    model_uri = "models:/hirewise-inflation-detector@champion"

    # MLflow scikit-learn flavor dwara serialized champion model memory mein load kar rahe hain
    return mlflow.sklearn.load_model(model_uri)

def create_app(model_loader=load_champion_model):
    """
    Application-factory pattern: unit testing ke dauran FakeModel inject karne ki suvidha deta hai.
    """
    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        # Startup event: model ko ek hi baar load karke app state mein store karte hain
        try:
            app_instance.state.model = model_loader()
            print("[Lifespan] Champion model loaded successfully.")
        except Exception as err:
            app_instance.state.model = None
            print(f"[Lifespan Warning] Could not load model at startup: {err}")
        yield
        # Shutdown event (if any cleanup is needed)

    app = FastAPI(
        title="HireWise Resume Inflation Detection API",
        version="1.0.0",
        lifespan=lifespan
    )

    @app.get("/health")
    def health():
        """Health check endpoint to verify service availability"""
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        """Predicts whether the candidate's claimed skill on resume is inflated"""
        model = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model is currently unavailable or not loaded.")

        diff = round(req.claimed_skill - req.measured_ability, 2)
        features = [[req.claimed_skill, req.measured_ability, diff]]

        prediction = model.predict(features)
        is_inflated = bool(prediction[0] == 1)

        return PredictResponse(
            inflated=is_inflated,
            difference=diff
        )

    return app

# Module-level instance for uvicorn runtime: uvicorn serve:app
app = create_app()
