# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

logger = logging.getLogger("LLMClient")
logger.setLevel(logging.INFO)

MODEL_REGISTRY = {
    "gpt": "gpt-5",
    "gpt-5": "gpt-5",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gemini": "gemini-3",
    "gemini-3": "gemini-3",
    "gemini-2.0": "gemini-2.0-flash-exp",
    "gemini-1.5": "gemini-1.5-pro-latest",
    "qwen-vl": "local",
}


@dataclass
class LLMOutput:
    text: str
    raw_response: Dict[str, Any]
    model_used: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None


def _get_api_key(provider: str) -> str:
    key = os.getenv(f"{provider.upper()}_API_KEY")
    if not key:
        logger.warning(f"[{provider}] API Key not found in ENV. Calls may fail.")
        return ""
    return key


def _call_openai_backend(model: str, messages: List[Dict], config: Dict) -> LLMOutput:
    try:
        from openai import OpenAI, RateLimitError, APIError
    except ImportError:
        raise ImportError("Missing dependency: pip install openai")

    client = OpenAI(api_key=_get_api_key("OPENAI"))

    retries = config.get("max_retries", 3)
    backoff = 2

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.get("temperature", 0.2),
                max_tokens=config.get("max_tokens", 2048),
                top_p=config.get("top_p", 0.95),
                response_format={"type": "json_object"} if config.get("json_mode") else None,
            )

            choice = response.choices[0]
            return LLMOutput(
                text=choice.message.content,
                raw_response=response.model_dump(),
                model_used=response.model,
                usage=response.usage.model_dump() if response.usage else {},
                finish_reason=choice.finish_reason,
            )

        except RateLimitError:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except APIError as e:
            logger.error(f"[OpenAI] API Error: {e}")
            if attempt < retries:
                time.sleep(1)
                continue
            raise


def _call_gemini_backend(model: str, messages: List[Dict], config: Dict) -> LLMOutput:
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
    except ImportError:
        raise ImportError("Missing dependency: pip install google-generativeai")

    genai.configure(api_key=_get_api_key("GOOGLE"))

    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    gen_config = genai.types.GenerationConfig(
        temperature=config.get("temperature", 0.2),
        max_output_tokens=config.get("max_tokens", 4096),
        top_p=config.get("top_p", 0.95),
    )

    system_instruction = None
    chat_history = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_instruction = content
        elif role == "user":
            chat_history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            chat_history.append({"role": "model", "parts": [content]})

    gemini_model = genai.GenerativeModel(model_name=model, system_instruction=system_instruction)

    last_user_msg = chat_history.pop() if chat_history and chat_history[-1]["role"] == "user" else None
    if not last_user_msg:
        raise ValueError("[Gemini] User message missing from end of prompt.")

    retries = config.get("max_retries", 3)
    for attempt in range(retries + 1):
        try:
            if chat_history:
                chat = gemini_model.start_chat(history=chat_history)
                response = chat.send_message(
                    last_user_msg["parts"][0],
                    generation_config=gen_config,
                    safety_settings=safety_settings,
                )
            else:
                response = gemini_model.generate_content(
                    last_user_msg["parts"][0],
                    generation_config=gen_config,
                    safety_settings=safety_settings,
                )

            return LLMOutput(
                text=response.text,
                raw_response={"feedback": str(response.prompt_feedback)},
                model_used=model,
                usage={},
                finish_reason="stop",
            )
        except Exception as e:
            logger.warning(f"[Gemini] Error: {e}. Retry {attempt}/{retries}")
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise


def simple_chat(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    json_mode: bool = False,
    **kwargs,
) -> LLMOutput:
    resolved_model = MODEL_REGISTRY.get(model, model)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    config = kwargs
    config["json_mode"] = json_mode

    if "gemini" in resolved_model.lower():
        return _call_gemini_backend(resolved_model, messages, config)
    if "gpt" in resolved_model.lower():
        return _call_openai_backend(resolved_model, messages, config)

    return _call_openai_backend(resolved_model, messages, config)


def call_local_vl_qwen(
    model_dir: str,
    system: str,
    user: str,
    images: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> LLMOutput:
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise RuntimeError("Missing local inference libs: transformers, qwen_vl_utils, torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.bfloat16
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=None, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    except Exception as e:
        logger.error(f"[LocalVL] Model load failed: {e}")
        return LLMOutput(text="", raw_response={"error": str(e)}, model_used="local-qwen", usage={})

    model = model.to(device).eval()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": []},
    ]

    for img_path in images:
        if os.path.exists(img_path):
            messages[1]["content"].append({"type": "image", "image": img_path})
    messages[1]["content"].append({"type": "text", "text": user})

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
        )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return LLMOutput(
        text=output_text,
        raw_response={"backend": "transformers_local"},
        model_used="qwen2.5-vl-local",
        usage={"images": len(images)},
    )


if __name__ == "__main__":
    print("=== LLM Client Diagnostic ===")

    print("\n[Test 1] OpenAI (GPT-5)")
    if os.getenv("OPENAI_API_KEY"):
        try:
            resp = simple_chat("gpt", "Hello JSON", system="Reply in JSON", json_mode=True)
            print(f"Response: {resp.text}")
        except Exception as e:
            print(f"Fail: {e}")
    else:
        print("Skipped (No Key)")

    print("\n[Test 2] Google (Gemini 3)")
    if os.getenv("GOOGLE_API_KEY"):
        try:
            resp = simple_chat("gemini", "Explain quantum physics in 1 sentence.")
            print(f"Response: {resp.text}")
        except Exception as e:
            print(f"Fail: {e}")
    else:
        print("Skipped (No Key)")

    print("\n[Test 3] Local GPU (Qwen-VL)")
    local_path = "sample_model_dir"
    if os.path.exists(local_path):
        try:
            resp = call_local_vl_qwen(local_path, "You are a scanner.", "Read text.", [])
            print(f"Response: {resp.text}")
        except Exception as e:
            print(f"Fail: {e}")
    else:
        print(f"Skipped (Path not found: {local_path})")
