# Week 2 AI Chat · Café Aurora

Script que le pregunta a un LLM (LM Studio local u OpenAI) y extrae datos de reseñas
con salidas estructuradas. Es la parada 5 del asistente de Café Aurora, por sí sola.

## Instalación

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
```

## Configuración

- **LM Studio (gratis):** deja `USE_LOCAL_LLM=true`, carga `llama-3.2-3b-instruct` y enciende el servidor.
- **OpenAI (con la key del curso):** pon `USE_LOCAL_LLM=false` y la key que te dio tu instructor en `OPENAI_API_KEY`.

## Uso

```bash
python week2_ai_chat.py            # pregunta y respuesta, con tokens y costo
python week2_ai_chat.py extract    # reseña a ReviewData validado
python raw_call.py                 # la misma llamada con requests, sin SDK
```

## Decisiones

- Un solo cliente `OpenAI` para los dos proveedores: solo cambian `base_url` y `api_key`.
- Errores capturados por clase del SDK; el SDK reintenta 429, 5xx y conexión, nunca 401.
- Costo con precio de entrada y de salida por separado (gpt-4o-mini, verificado 2026-09-25).
- Salida estructurada con `.parse()` y Pydantic; si el servidor rechaza `json_schema`, schema en el prompt y validación propia.