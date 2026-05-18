import os
import sys

# Force UTF-8 output on Windows to avoid charmap encoding errors with Unicode symbols
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import tempfile
import base64
import requests
import asyncio
import json
import re
import hashlib
from datetime import date
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
from pathlib import Path

from modules.preprocessors import preprocess
from modules.physics_analyzer import analyze_physics

app = FastAPI(title="RePrompt API")

os.makedirs("static", exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDDHs-KwdFJMbjRYIgmyW4KAeFgl6FDMpM")

# It's good practice to add a safeguard so the app fails loudly if the key is missing
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set!")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# Ordered list of models to try (each has its own separate free-tier quota)
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",   # newest, separate quota
    "gemini-2.0-flash-lite",   # lightweight, generous quota
    "gemini-2.5-flash",        # very capable, separate quota
    "gemini-2.0-flash",        # original
]

import time


def _extract_text_from_gemini(response_json: dict) -> str:
    """Safely extract the text content from a Gemini API response.
    Handles multiple response structures including thinking models with multi-part outputs."""
    try:
        candidates = response_json.get("candidates", [])
        if not candidates:
            # Check if blocked by safety
            block_reason = response_json.get("promptFeedback", {}).get("blockReason", "")
            if block_reason:
                raise ValueError(f"Gemini blocked the request: {block_reason}")
            raise ValueError("Gemini returned no candidates")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "SAFETY":
            raise ValueError("Gemini blocked output due to safety filters")

        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            raise ValueError("Gemini returned empty parts")

        # Collect text from all parts (some models return thinking + output)
        text_parts = [p["text"] for p in parts if "text" in p]
        if not text_parts:
            raise ValueError("No text found in Gemini response parts")

        # Return the LAST text part (thinking models put the answer last)
        return text_parts[-1]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected Gemini response structure: {e}")


def _extract_json_from_text(text: str) -> dict | list:
    """Robustly extract JSON from Gemini output text.
    Handles: bare JSON, ```json fences, ``` fences, extra surrounding text."""
    text = text.strip()

    # 1. Try parsing the whole text as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from markdown code fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Try to find the outermost JSON object { ... } or array [ ... ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        # Find matching closing bracket by counting nesting
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start_idx:i + 1])
                except json.JSONDecodeError:
                    break

    raise ValueError(f"Could not extract valid JSON from Gemini output. Raw text (first 500 chars): {text[:500]}")

def _gemini_post(payload: dict) -> requests.Response:
    """Try each model in order; retry once after a short delay on 429."""
    last_resp = None
    for model in GEMINI_MODELS:
        url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
        print(f"[RePrompt] Trying model: {model}...")
        resp = requests.post(url, json=payload, timeout=30)
        last_resp = resp
        if resp.status_code == 200:
            print(f"[RePrompt] [OK] Success with {model}")
            return resp
        if resp.status_code == 429:
            # Try a quick retry for this model
            time.sleep(5)
            resp = requests.post(url, json=payload, timeout=30)
            last_resp = resp
            if resp.status_code == 200:
                return resp
            print(f"[RePrompt] 429 quota exhausted on {model}, trying next...")
            continue
        if resp.status_code in (401, 403):
            # Auth failure — no point trying other models with same key
            print(f"[RePrompt] Auth error on {model}: {resp.text[:300]}")
            return resp
        # Any other error (400, 500, etc.) — log it and try next model
        print(f"[RePrompt] Error {resp.status_code} on {model}: {resp.text[:300]}")
        continue
    return last_resp  # All models exhausted, return last response

async def generate_prompt_from_gemini(image_path: str, physics_stats: dict) -> dict:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    ext = Path(image_path).suffix.lower()
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.bmp': 'image/bmp'}
    mime_type = mime_map.get(ext, 'image/jpeg')
    
    llm_prompt = (
        "You are an elite AI image prompt engineer. Extract an exhaustively detailed prompt from the provided image for Midjourney v6 or Stable Diffusion.\n"
        "Detect every nuance, art style, lighting effect, and character detail. Format as flowing descriptive paragraphs.\n"
        "IMPORTANT: Keep your response concise. The prompt should be under 300 words and the negative prompt under 100 words.\n\n"
        f"Incorporate these verified physical measurements naturally:\n"
        f"- Brightness: {physics_stats['brightness_class']} (Mean: {physics_stats['mean_brightness']})\n"
        f"- Depth of Field: {physics_stats['dof_class']} focus\n"
        f"- Shadows: {physics_stats['shadow_hardness']}\n"
        f"- Lighting Direction: {physics_stats['light_direction']}\n"
        f"- Contrast Ratio: {physics_stats['contrast_ratio']}\n\n"
        "Additionally, generate a negative prompt listing things to avoid/exclude.\n"
        "Output ONLY valid JSON in the following format (no markdown, no backticks):\n"
        '{\n'
        '  "prompt": "The positive prompt text here...",\n'
        '  "negative_prompt": "The negative prompt text here..."\n'
        '}'
    )
    
    payload = {
        "contents": [{"parts": [
            {"text": llm_prompt},
            {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        ]}],
        "generationConfig": {"temperature": 0.15, "topP": 0.85, "maxOutputTokens": 4096}
    }
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: _gemini_post(payload))
        if response.status_code == 200:
            result = response.json()
            try:
                text = _extract_text_from_gemini(result)
                print(f"[RePrompt] Raw Gemini output (first 500 chars): {text[:500]}")
                try:
                    parsed = _extract_json_from_text(text)
                except ValueError:
                    # Fallback: try to salvage truncated JSON by extracting fields with regex
                    print("[RePrompt] JSON parse failed, attempting regex salvage...")
                    prompt_match = re.search(r'"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"?', text, re.DOTALL)
                    neg_match = re.search(r'"negative_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"?', text, re.DOTALL)
                    if prompt_match:
                        parsed = {
                            "prompt": prompt_match.group(1).replace('\\n', ' ').replace('\\"', '"'),
                            "negative_prompt": neg_match.group(1).replace('\\n', ' ').replace('\\"', '"') if neg_match else ""
                        }
                    else:
                        # Last resort: use the raw text as the prompt
                        parsed = {"prompt": text.strip(), "negative_prompt": ""}
                # Ensure it's a dict with the expected keys
                if isinstance(parsed, dict):
                    return {
                        "prompt": parsed.get("prompt", ""),
                        "negative_prompt": parsed.get("negative_prompt", "")
                    }
                else:
                    raise ValueError("Expected JSON object, got array")
            except ValueError as e:
                print(f"[RePrompt] JSON extraction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")
        elif response.status_code == 429:
            raise HTTPException(status_code=429, detail="Gemini API is currently overloaded. Please try again in 10 seconds.")
        else:
            raise HTTPException(status_code=response.status_code, detail="Gemini API request failed. Check API key and quota.")
    except requests.exceptions.RequestException as e:
        print(f"Gemini API error: {e}")
        raise HTTPException(status_code=503, detail="Could not connect to Gemini API. Please check your internet connection.")

class ImproveRequest(BaseModel):
    text: str

class AnatomyRequest(BaseModel):
    prompt: str

class DailySubmitRequest(BaseModel):
    challenge_id: str
    user_prompt: str
    user_negative_prompt: Optional[str] = ""

# ── Daily Challenge Data ──────────────────────────────────────────────────────
DAILY_CHALLENGES = [
    {"image_url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&q=80", "category": "Portrait", "difficulty": 2},
    {"image_url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80", "category": "Landscape", "difficulty": 1},
    {"image_url": "https://images.unsplash.com/photo-1486325212027-8081e485255e?w=800&q=80", "category": "Architecture", "difficulty": 2},
    {"image_url": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800&q=80", "category": "Food Photography", "difficulty": 1},
    {"image_url": "https://images.unsplash.com/photo-1474511320723-9a56873571b7?w=800&q=80", "category": "Wildlife", "difficulty": 3},
    {"image_url": "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=800&q=80", "category": "Abstract", "difficulty": 3},
    {"image_url": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800&q=80", "category": "Cityscape", "difficulty": 2},
    {"image_url": "https://images.unsplash.com/photo-1518173946687-a53f45400867?w=800&q=80", "category": "Nature Macro", "difficulty": 2},
    {"image_url": "https://images.unsplash.com/photo-1513542789411-b6a5d4f31634?w=800&q=80", "category": "Moody Landscape", "difficulty": 2},
    {"image_url": "https://images.unsplash.com/photo-1519608487953-e999c86e7455?w=800&q=80", "category": "Night Photography", "difficulty": 3},
    {"image_url": "https://images.unsplash.com/photo-1495567720989-cebdbdd97913?w=800&q=80", "category": "Minimalist", "difficulty": 1},
    {"image_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80", "category": "Portrait Close-up", "difficulty": 2},
    {"image_url": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800&q=80", "category": "Forest", "difficulty": 1},
    {"image_url": "https://images.unsplash.com/photo-1551244072-5d12893278ab?w=800&q=80", "category": "Product Shot", "difficulty": 2},
]

# Cache for daily challenge analysis (avoid re-processing same image)
_daily_cache = {}

@app.post("/api/improve")
async def improve_text(request: ImproveRequest):
    prompt_template = (
        "You are an expert AI prompt engineer. Rewrite and vastly improve the following basic image generation prompt "
        "by adding descriptive details, lighting, camera angles, and art style, while maintaining the core subject.\n\n"
        f"Original Prompt: {request.text}\n\n"
        "Output ONLY the improved prompt text."
    )
    
    payload = {
        "contents": [{"parts": [{"text": prompt_template}]}],
        "generationConfig": {"temperature": 0.5, "topP": 0.9, "maxOutputTokens": 500}
    }
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: _gemini_post(payload))
        if response.status_code == 200:
            result = response.json()
            try:
                improved = result["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                improved = "No response from model."
            return JSONResponse(content={"result": improved.strip()})
        else:
            return JSONResponse(content={"result": f"Model error: {response.text}"}, status_code=500)
    except Exception as e:
        print(f"Error improving text: {e}")
        return JSONResponse(content={"result": "Backend failed to process text."}, status_code=500)

@app.post("/api/reprompt")
async def create_reprompt(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name
        
    try:
        meta = preprocess(temp_path)
        physics = analyze_physics(meta)
        
        stats = {
            "aspect_ratio": meta.aspect_ratio,
            "mean_brightness_global": meta.mean_brightness,
            "global_contrast": meta.global_contrast,
            "dominant_hues": meta.dominant_hues,
            "brightness_class": physics.brightness_class,
            "mean_brightness": physics.mean_brightness,
            "dof_class": physics.dof_class,
            "sharpness_score": physics.sharpness_score,
            "shadow_hardness": physics.shadow_hardness,
            "shadow_score": physics.shadow_score,
            "light_direction": physics.light_direction,
            "contrast_ratio": physics.contrast_ratio,
        }
        
        prompt_data = await generate_prompt_from_gemini(temp_path, stats)
        
        return JSONResponse(content={
            "reprompt": prompt_data.get("prompt", ""),
            "negative_prompt": prompt_data.get("negative_prompt", ""),
            "stats": stats
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Learning Feature Endpoints ────────────────────────────────────────────────

@app.post("/api/anatomy")
async def analyze_anatomy(request: AnatomyRequest):
    """Parse a prompt into color-coded categorized segments."""
    anatomy_prompt = (
        "Parse the following AI image generation prompt into categorized segments. "
        "Each segment should be a meaningful phrase from the prompt.\n\n"
        f'Prompt: "{request.prompt}"\n\n'
        "Categorize each segment into one of these categories:\n"
        "- subject: Description of the main subject/scene\n"
        "- lighting: Lighting conditions, light quality, direction\n"
        "- composition: Camera angle, framing, perspective\n"
        "- style: Art style, medium, artistic technique\n"
        "- mood: Atmosphere, emotion, feeling\n"
        "- technical: Camera settings, lens, depth of field, resolution\n\n"
        "Return ONLY a JSON array (no markdown, no backticks):\n"
        '[{"text": "exact phrase from prompt", "category": "subject", '
        '"tooltip": "Brief explanation of why this element matters in prompts"}]'
    )
    payload = {
        "contents": [{"parts": [{"text": anatomy_prompt}]}],
        "generationConfig": {"temperature": 0.1, "topP": 0.8, "maxOutputTokens": 1500}
    }
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: _gemini_post(payload))
        if response.status_code == 200:
            result = response.json()
            raw_text = _extract_text_from_gemini(result)
            segments = _extract_json_from_text(raw_text)
            return JSONResponse(content={"segments": segments})
        else:
            raise HTTPException(status_code=response.status_code, detail="Anatomy analysis failed.")
    except json.JSONDecodeError:
        return JSONResponse(content={"segments": [{"text": request.prompt, "category": "subject", "tooltip": "Full prompt"}]})
    except Exception as e:
        print(f"Anatomy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_evaluation(image_path: str, user_prompt: str, user_negative_prompt: str = ""):
    """Shared logic: run physics analysis + Gemini scoring on an image."""
    meta = preprocess(image_path)
    physics = analyze_physics(meta)
    stats = {
        "brightness_class": physics.brightness_class,
        "mean_brightness": physics.mean_brightness,
        "dof_class": physics.dof_class,
        "sharpness_score": physics.sharpness_score,
        "shadow_hardness": physics.shadow_hardness,
        "shadow_score": physics.shadow_score,
        "light_direction": physics.light_direction,
        "contrast_ratio": physics.contrast_ratio,
    }

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lower()
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.bmp': 'image/bmp'}
    mime_type = mime_map.get(ext, 'image/jpeg')

    eval_prompt = (
        "You are an expert prompt engineering instructor scoring a student's attempt.\n\n"
        f"Physical measurements of this image:\n"
        f"- Brightness: {stats['brightness_class']} (Mean: {stats['mean_brightness']})\n"
        f"- Depth of Field: {stats['dof_class']}\n"
        f"- Shadows: {stats['shadow_hardness']}\n"
        f"- Light Direction: {stats['light_direction']}\n"
        f"- Contrast Ratio: {stats['contrast_ratio']}\n\n"
        f'Student\'s prompt attempt: "{user_prompt}"\n'
        f'Student\'s negative prompt attempt: "{user_negative_prompt}"\n\n'
        "Score the student 1-10 and analyze their prompt. Return ONLY valid JSON (no markdown, no backticks):\n"
        '{"score": 7, "feedback": "Overall feedback", '
        '"ideal_prompt": "The ideal prompt for this image", '
        '"ideal_negative_prompt": "The ideal negative prompt for this image (what to avoid)", '
        '"breakdown": ['
        '{"element": "Subject Description", "status": "covered", "detail": "..."},'
        '{"element": "Lighting", "status": "missing", "detail": "..."},'
        '{"element": "Composition", "status": "partial", "detail": "..."},'
        '{"element": "Style/Medium", "status": "covered", "detail": "..."},'
        '{"element": "Mood/Atmosphere", "status": "missing", "detail": "..."},'
        '{"element": "Technical Details", "status": "wrong", "detail": "..."},'
        '{"element": "Shadow Description", "status": "missing", "detail": "..."},'
        '{"element": "Negative Prompt Analysis", "status": "missing", "detail": "Feedback on their negative prompt"},'
        '{"element": "Color Palette", "status": "missing", "detail": "..."}]}\n'
        "Use status values: covered, missing, partial, wrong. Be encouraging but honest."
    )
    payload = {
        "contents": [{"parts": [
            {"text": eval_prompt},
            {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        ]}],
        "generationConfig": {"temperature": 0.15, "topP": 0.85, "maxOutputTokens": 4096}
    }
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: _gemini_post(payload))
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Evaluation failed.")
    raw_text = _extract_text_from_gemini(response.json())
    print(f"[RePrompt] Raw evaluation output (first 300 chars): {raw_text[:300]}")
    evaluation = _extract_json_from_text(raw_text)
    evaluation["stats"] = stats
    return evaluation


@app.post("/api/evaluate")
async def evaluate_prompt(
    file: UploadFile = File(...), 
    user_prompt: str = Form(...),
    user_negative_prompt: str = Form(default="")
):
    """Score a user's prompt attempt against an uploaded image."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name
    try:
        result = await _run_evaluation(temp_path, user_prompt, user_negative_prompt)
        return JSONResponse(content=result)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="Failed to parse AI evaluation response.")
    except Exception as e:
        print(f"Evaluate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/daily-challenge")
async def get_daily_challenge():
    """Return today's daily challenge."""
    today = date.today()
    day_index = today.timetuple().tm_yday % len(DAILY_CHALLENGES)
    challenge = DAILY_CHALLENGES[day_index]
    return JSONResponse(content={
        "id": today.isoformat(),
        "image_url": challenge["image_url"],
        "category": challenge["category"],
        "difficulty": challenge["difficulty"],
    })


@app.post("/api/daily-evaluate")
async def evaluate_daily(request: DailySubmitRequest):
    """Evaluate a user's prompt for the daily challenge."""
    today = date.today()
    day_index = today.timetuple().tm_yday % len(DAILY_CHALLENGES)
    challenge = DAILY_CHALLENGES[day_index]
    image_url = challenge["image_url"]

    # Download the challenge image
    try:
        img_resp = requests.get(image_url, timeout=15)
        img_resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download challenge image: {e}")

    ext = ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(img_resp.content)
        temp_path = tmp.name
    try:
        user_negative_prompt = getattr(request, 'user_negative_prompt', "")
        result = await _run_evaluation(temp_path, request.user_prompt, user_negative_prompt)
        return JSONResponse(content=result)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI evaluation response.")
    except Exception as e:
        print(f"Daily evaluate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
