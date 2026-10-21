from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from text_generator import TextGenerator
app = FastAPI(title = "Shakespeare GPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

generator = TextGenerator()
class GenerateRequest(BaseModel):
  prompt: str = Field(min_length = 1, max_length = 2000)
  max_new_tokens: int = Field(default = 100, ge = 1, le = 200)
  temperature: float = Field(default=0.8, gt=0, le=2.0)


@app.get("/")
def home():
  return {"message": "Shakespeare GPT API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    try:
        result = generator.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        return {"generated_text": result}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Text generation failed",
        ) from exc
