from fastapi import FastAPI

from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Math Physics AI Tutor",
        version="0.1.0",
        description="A production-ready tutor for mathematics and physics reasoning.",
    )
    app.include_router(router)
    return app


app = create_app()
