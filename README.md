# Shakespeare GPT

A full-stack text-generation application powered by a custom GPT-style Transformer trained on Shakespeare text.

## Project Structure

```
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
│   └── requirements.txt
├── shakespeare-frontend/
│   ├── src/
│   ├── package.json
│   └── ...
└── README.md
```

## Prerequisites

* Python 3.13
* Node.js and npm
* The trained model weights and tokenizer artifacts

## Backend Setup

Open PowerShell and run:

```
cd "C:\Users\Hp\OneDrive\Desktop\model deploy\backend"

py -3.13 -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the API server:

```
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The API documentation is available at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Health check:

[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## Frontend Setup

Open a second PowerShell terminal:

```
cd "C:\Users\Hp\OneDrive\Desktop\model deploy\shakespeare-frontend"

npm.cmd install

npm.cmd run dev
```

Open the local Vite URL printed in the terminal, normally:

[http://localhost:5173](http://localhost:5173/)

## API

### POST `<span>/generate</span>`

Example request:

```
{
  "prompt": "ROMEO: What light through yonder window breaks?",
  "max_new_tokens": 100,
  "temperature": 0.8
}
```

The response contains a `<span>generated_text</span>` field.

## Model

The application uses a decoder-only Transformer with causal self-attention, token embeddings, positional embeddings, residual connections, layer normalization, and a language-model head.

The tokenizer and model configuration must match the trained checkpoint.

## Current Limitations

* The model is small and produces limited-quality Shakespeare-style text.
* The backend runs locally and is not publicly deployed.
* The frontend requires the backend to be running for generation.

## Development Notes

Run the frontend and backend in separate terminals. Keep the backend's API URL and CORS configuration consistent with the frontend development origin.
