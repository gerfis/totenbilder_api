from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

# Globale Imports verzögern wir ggf. nicht, da sie eh cached sind, 
# aber wir nutzen die Router direkt
from index import router as upload_router, get_model as get_img_model
from search import router as search_router, get_model as get_text_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload Models in Background to save startup time?
    # Oder einfach lazy laden lassen bei erstem Request.
    # Da RAM sparen gewünscht ist, laden wir sie vielleicht besser lazy,
    # aber um "warm" zu sein, triggern wir es an.
    
    print("Preloading Models...")
    # Wir laden sie einfach einmal an, damit sie im RAM sind
    # Das OS wird dank Torch Shared Memory nutzen wo möglich, 
    # aber es sind 2 verschiedene Modelle.
    get_img_model()
    get_text_model()
    print("Models preloaded.")
    yield
    print("Shutdown.")

app = FastAPI(lifespan=lifespan, title="Totenbilder API")

# Configure CORS
origins = [
    "http://localhost:3000",
    "https://totenbilder.at",
    "https://www.totenbilder.at",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(upload_router, prefix="/api", tags=["Upload/Index"])
app.include_router(search_router, prefix="/api", tags=["Search"])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Port 8000 als Standard
    uvicorn.run(app, host="0.0.0.0", port=8000)