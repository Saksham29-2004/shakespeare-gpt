# Shakespeare GPT

A full-stack text-generation application powered by a custom GPT-style Transformer trained on Shakespeare text.

## Project Structure

```text
model deploy/
├── backend/
│   ├── api.py
│   ├── model.py
│   ├── loader.py
│   ├── tokenizer.py
│   ├── inference.py
│   ├── text_generator.py
│   ├── config.json
│   ├── tokenizer.json
│   ├── shakespeare_gpt.pth
│   ├── requirements.txt
│   └── test_api.py
├── shakespeare-frontend/
│   ├── src/
│   ├── package.json
│   └── .env.local
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.13
- Node.js and npm
- The trained model checkpoint, configuration, and tokenizer files

## Backend Setup

Open PowerShell in the project root and run:

```powershell
cd .\backend
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the API server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Frontend Setup

Open a second PowerShell terminal in the project root:

```powershell
cd .\shakespeare-frontend
npm.cmd install
```

Create a `.env.local` file in `shakespeare-frontend/` containing:

```dotenv
VITE_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```powershell
npm.cmd run dev
```

Open the local URL printed by Vite, normally http://localhost:5173.

The backend must be running before generating text. Restart the Vite development server after changing environment variables.

## Running Tests

From the project root:

```powershell
cd .\backend
.\.venv\Scripts\python.exe -m pytest -v
```

The tests cover API health, valid generation requests, input validation, and generation failure handling.

## API

### `POST /generate`

Example request:

```json
{
  "prompt": "ROMEO: What light through yonder window breaks?",
  "max_new_tokens": 100,
  "temperature": 0.8
}
```

The response contains a `generated_text` field.

Validation constraints:

- `prompt`: 1–2000 characters
- `max_new_tokens`: 1–200
- `temperature`: greater than 0 and at most 2.0

## Model

The application uses a decoder-only Transformer with causal self-attention, token embeddings, positional embeddings, residual connections, layer normalization, and a language-model head.

The model checkpoint, tokenizer, and configuration must correspond to the same trained model.

## Current Limitations

- The model is small and generates imperfect Shakespeare-style text.
- The backend currently runs locally and is not publicly deployed.
- Text generation requires the backend to be running.
- The frontend's configured API URL and backend CORS origins must match the deployment environment.

## Development Notes

Run the frontend and backend in separate terminals. Do not commit the Python virtual environment, frontend dependencies, or local environment files.
