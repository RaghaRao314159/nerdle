# Agent Server for Nerdle Game

Python backend service that loads a language model (e.g., Qwen 0.6B) and provides inference for the Nerdle agent mode.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python agent_server.py
   ```

   Or specify a custom model:
   ```bash
   MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" python agent_server.py
   ```

3. **Default configuration:**
   - Port: 5000 (configurable via `PORT` environment variable)
   - Model: Qwen/Qwen2.5-0.5B-Instruct (configurable via `MODEL_NAME`)

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /guess`
Generate next guess based on game history.

**Request:**
```json
{
  "history": [
    {
      "guess": "48-32=16",
      "feedback": ["present", "absent", "correct", ...]
    }
  ],
  "attemptNumber": 2
}
```

**Response:**
```json
{
  "guess": "10+20=30"
}
```

### `POST /load_model`
Load a different model.

**Request:**
```json
{
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model Qwen/Qwen2.5-0.5B-Instruct loaded successfully"
}
```

## Environment Variables

- `PORT`: Server port (default: 5000)
- `MODEL_NAME`: Model to load (default: "Qwen/Qwen2.5-0.5B-Instruct")

## Next.js Integration

The Next.js frontend calls this service via the `/app/api/agent/route.ts` proxy endpoint.

Make sure to set `AGENT_SERVER_URL` environment variable in your Next.js app if running on a different host/port:

```bash
AGENT_SERVER_URL="http://localhost:5000" npm run dev
```

## Model Requirements

- The model should support chat templates
- Recommended: Qwen2.5-0.5B-Instruct or similar small instruction-tuned models
- For Qwen 0.6B, use: `Qwen/Qwen2.5-0.5B-Instruct` (closest available equivalent)

## Troubleshooting

- If the model fails to load, the server will use fallback guess generation
- Check logs for model loading errors
- Ensure you have sufficient GPU/RAM for the model size
- For CPU-only inference, ensure you have enough RAM (models are loaded in float32 on CPU)

