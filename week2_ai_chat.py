"""
week2_ai_chat.py: la parada 5 del asistente de Café Aurora, por sí sola.

Funciona con LM Studio (local) u OpenAI (nube): USE_LOCAL_LLM en .env decide.
    python week2_ai_chat.py            haces una pregunta, recibes texto
    python week2_ai_chat.py extract    pegas una reseña, recibes ReviewData
"""

import json
import os
import sys
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Precios de gpt-4o-mini en USD por 1M de tokens, verificados el 2026-09-25.
# Los precios cambian: revisa https://developers.openai.com/api/docs/pricing
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60

SYSTEM_PROMPT = (
    "You are an assistant for the staff of Café Aurora, a coffee shop. "
    "Answer general questions accurately, in the language of the question, in at most three sentences. "
    "You have NO information about Café Aurora itself (menu, prices, hours, policies). "
    "If asked about it, say you do not have that information yet."
)

class ReviewData(BaseModel):
    """The data Café Aurora wants from each customer review."""

    product: str = Field(description="The product the review talks about")
    price_mxn: float | None = Field(default=None, description="The price in Mexican pesos, if it is mentioned")
    rating: int = Field(description="The rating from 1 to 5, where 5 is the best")

def load_config() -> dict[str, Any]:
    """Lee .env, lo valida y devuelve la configuración. Lanza ValueError."""
    load_dotenv()
    use_local = os.getenv("USE_LOCAL_LLM", "true").strip().lower() == "true"
    config: dict[str, Any] = {
        "use_local": use_local,
        "max_tokens": int(os.getenv("MAX_TOKENS", "150")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "timeout": float(os.getenv("TIMEOUT", "30")),
        "max_retries": int(os.getenv("MAX_RETRIES", "2")),
    }
    if use_local:
        config["base_url"] = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        config["api_key"] = "not-needed"
        config["model"] = os.getenv("MODEL", "llama-3.2-3b-instruct")
        config["provider"] = "LM Studio (local)"
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "USE_LOCAL_LLM=false pero OPENAI_API_KEY está vacía.\n"
                "  1. Pídele a tu instructor la key del curso\n"
                "  2. Agrégala a .env: OPENAI_API_KEY=sk-proj-...\n"
                "  O pon USE_LOCAL_LLM=true para usar LM Studio (gratis)."
            )
        config["base_url"] = None  # None = el default del SDK, https://api.openai.com/v1
        config["api_key"] = api_key
        config["model"] = os.getenv("MODEL", "gpt-4o-mini")
        config["provider"] = "OpenAI (nube)"
    return config

def create_client(config: dict[str, Any]) -> OpenAI:
    """El mismo cliente para los dos proveedores: solo cambian base_url y api_key."""
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=config["timeout"],
        max_retries=config["max_retries"],
    )
def check_server(client: OpenAI, config: dict[str, Any]) -> bool:
    """Solo LM Studio: GET /v1/models antes de preguntar nada."""
    try:
        model_ids = [model.id for model in client.models.list()]
    except openai.APIConnectionError:
        print(f"❌ LM Studio no responde en {config['base_url']}")
        print("💡 Abre LM Studio, carga el modelo y enciende el servidor (pestaña Developer).")
        return False
    # Con varias variantes del mismo modelo, LM Studio agrega "@variante" al id.
    base_ids = {model_id.split("@")[0] for model_id in model_ids}
    if config["model"] not in model_ids and config["model"] not in base_ids:
        print(f"❌ El modelo '{config['model']}' no está disponible en LM Studio.")
        print(f"💡 Disponibles: {', '.join(model_ids) or '(ninguno)'}. Ajusta MODEL en .env.")
        return False

    print(f"✅ LM Studio está encendido con {config['model']}")
    return True

def call_llm(
    client: OpenAI,
    config: dict[str, Any],
    messages: list[dict[str, str]],
    schema: type[BaseModel] | None = None,
) -> Any:
    """Una llamada al modelo. Devuelve la respuesta del SDK, o None tras explicar el error."""
    try:
        if schema is None:
            return client.chat.completions.create(
                model=config["model"],
                messages=messages,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
        return client.chat.completions.parse(
            model=config["model"],
            messages=messages,
            response_format=schema,
            temperature=0,
            max_tokens=config["max_tokens"],
        )
    except openai.APITimeoutError:
        print(f"❌ Sin respuesta tras {config['timeout']:g} s (y {config['max_retries']} reintentos).")
        print("💡 Local: cierra otras apps o usa un modelo más chico. Nube: intenta más tarde.")
    except openai.APIConnectionError:
        print(f"❌ No se pudo conectar con {config['provider']}.")
        print("💡 Local: ¿está encendido el servidor de LM Studio? Nube: revisa tu internet.")
    except openai.AuthenticationError:
        print("❌ 401: la API key fue rechazada. Reintentar no lo arregla.")
        print("💡 Copia de nuevo la key en .env (sin comillas ni espacios) o pídele otra a tu instructor.")
    except openai.RateLimitError as error:
        print(f"❌ 429 de {config['provider']}: {error.code or 'rate limit'}.")
        print("💡 Espera un minuto. Si el código es insufficient_quota, la key llegó a su límite: avísale a tu instructor.")
    except openai.BadRequestError as error:
        if schema is not None:
            raise  # extract_review() tiene un plan B para este caso
        print(f"❌ 400: el servidor rechazó la petición: {error.message}")
    except openai.APIStatusError as error:
        print(f"❌ HTTP {error.status_code} de {config['provider']}.")
        if error.status_code >= 500:
            print(f"💡 Problema del servidor; el SDK ya reintentó {config['max_retries']} veces. Intenta más tarde.")
        else:
            print(f"💡 {error.message}")
    return None

def parse_response(response: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    """Convierte la respuesta del SDK en los campos que mostramos. No confía en content."""
    if not response.choices:
        print("❌ La respuesta llegó sin choices.")
        return None
    choice = response.choices[0]
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    if config["use_local"]:
        cost = 0.0
    else:
        cost = (prompt_tokens * PRICE_INPUT_PER_M + completion_tokens * PRICE_OUTPUT_PER_M) / 1_000_000
    result = {
        "content": choice.message.content,  # puede ser None
        "finish_reason": choice.finish_reason,
        "model": response.model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
        "warning": None,
    }
    if choice.finish_reason == "length":
        result["warning"] = f"Cortada en MAX_TOKENS={config['max_tokens']}: la respuesta está incompleta."
    elif choice.finish_reason == "content_filter":
        result["warning"] = "El filtro de contenido del proveedor detuvo la respuesta."
    elif choice.finish_reason != "stop":
        result["warning"] = f"finish_reason inesperado: {choice.finish_reason}"
    return result

def display_result(result: dict[str, Any], config: dict[str, Any]) -> None:
    """Imprime la respuesta y sus metadatos."""
    print("\n" + "=" * 60)
    print(result["content"] or "(el modelo no devolvió texto)")
    print("=" * 60)
    cost = "$0 (local)" if config["use_local"] else f"${result['cost']:.6f}"
    print(f"📊 {result['model']} · tokens entrada {result['prompt_tokens']} / salida {result['completion_tokens']} · costo {cost}")
    print(f"   finish_reason: {result['finish_reason']}")
    if result["warning"]:
        print(f"⚠️  {result['warning']}")

def extract_with_prompt(client: OpenAI, config: dict[str, Any], review: str) -> ReviewData | None:
    """Plan B para servidores sin json_schema: schema en el prompt, validamos nosotros."""
    schema = json.dumps(ReviewData.model_json_schema())
    messages = [
        {"role": "system", "content": f"Extract the review data. Reply ONLY with JSON matching this schema: {schema}"},
        {"role": "user", "content": review},
    ]
    response = call_llm(client, config, messages)
    if response is None or not response.choices:
        return None
    text = (response.choices[0].message.content or "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return ReviewData.model_validate_json(text)
    except ValidationError as error:
        print(f"❌ El JSON del modelo no cumple ReviewData: {error.errors()[0]['msg']}")
        return None

def extract_review(client: OpenAI, config: dict[str, Any], review: str) -> ReviewData | None:
    """Plan A: structured outputs con .parse(). Plan B si el servidor rechaza json_schema."""
    messages = [
        {"role": "system", "content": "Extract the data from this Café Aurora customer review."},
        {"role": "user", "content": review},
    ]
    try:
        completion = call_llm(client, config, messages, schema=ReviewData)
    except openai.BadRequestError:
        print("⚠️  Este servidor no acepta json_schema: va el schema en el prompt.")
        return extract_with_prompt(client, config, review)
    except openai.LengthFinishReasonError:
        print(f"❌ El JSON se cortó en MAX_TOKENS={config['max_tokens']}. Súbelo en .env.")
        return None
    except openai.ContentFilterFinishReasonError:
        print("❌ El filtro de contenido del proveedor detuvo la respuesta.")
        return None
    except ValidationError as error:
        print(f"❌ El JSON del modelo no cumple ReviewData: {error.errors()[0]['msg']}")
        return None
    if completion is None:
        return None
    message = completion.choices[0].message
    if message.refusal:
        print(f"❌ El modelo se negó: {message.refusal}")
        return None
    return message.parsed

def run_chat(client: OpenAI, config: dict[str, Any]) -> int:
    """Modo chat: una pregunta, una respuesta."""
    question = input("💬 Tu pregunta: ").strip()
    if not question:
        print("❌ Pregunta vacía: no hay nada que enviar.")
        return 1
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    print(f"🔄 Llamando a {config['provider']}...")
    response = call_llm(client, config, messages)
    if response is None:
        return 1
    result = parse_response(response, config)
    if result is None:
        return 1
    display_result(result, config)
    return 0

def run_extract(client: OpenAI, config: dict[str, Any]) -> int:
    """Modo extract: entra una reseña, sale un ReviewData."""
    review = input("📝 Pega una reseña: ").strip()
    if not review:
        print("❌ Reseña vacía: no hay nada que enviar.")
        return 1
    print(f"🔄 Extrayendo con {config['provider']}...")
    data = extract_review(client, config, review)
    if data is None:
        return 1
    print(f"✅ {type(data).__name__}: {data.model_dump()}")
    print(f"   price_mxn es {type(data.price_mxn).__name__}, rating es {type(data.rating).__name__}")
    return 0

def main() -> int:
    """Carga la config, verifica el servidor y corre el modo pedido."""
    print("🚀 Week 2 AI Chat · Café Aurora")
    try:
        config = load_config()
    except ValueError as error:
        print(f"❌ Error de configuración:\n{error}")
        return 1
    print(f"✅ Proveedor: {config['provider']} · modelo: {config['model']}")
    client = create_client(config)
    if config["use_local"] and not check_server(client, config):
        return 1
    if sys.argv[1:] == ["extract"]:
        return run_extract(client, config)
    return run_chat(client, config)

if __name__ == "__main__":
    sys.exit(main())