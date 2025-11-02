#!/usr/bin/env python3
"""
Agent Server for Nerdle Game
Loads a language model (Qwen 0.6B) and provides inference endpoint
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import torch at module level
try:
    import torch
except ImportError:
    torch = None
    logging.warning("PyTorch not available. Model inference will use fallback.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Global model variable
model = None
tokenizer = None
# torch should be available from module-level import above

def load_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Load the language model for inference.
    Adjust model_name based on available models.
    """
    global model, tokenizer
    
    try:
        if torch is None:
            raise ImportError("PyTorch is required but not installed. Please install with: pip install torch")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Loading model: {model_name}")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def generate_guess(history: List[Dict[str, Any]], attempt_number: int) -> tuple[str, str, dict]:
    """
    Generate the next Nerdle guess based on game history.
    
    Args:
        history: List of previous guesses with feedback
        attempt_number: Current attempt number (1-based)
    
    Returns:
        Tuple of (guess, source, metadata) where:
        - guess: Next guess as an 8-character equation string
        - source: "llm" or "fallback" 
        - metadata: Additional info (retry count, raw output, etc.)
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        logger.warning("⚠️  MODEL NOT LOADED - Using fallback")
        fallback_guess = generate_valid_fallback(history)
        return fallback_guess, "fallback", {"reason": "model_not_loaded"}
    
    # torch should be available from module-level import
    if torch is None:
        logger.warning("⚠️  PYTORCH NOT AVAILABLE - Using fallback")
        fallback_guess = generate_valid_fallback(history)
        return fallback_guess, "fallback", {"reason": "pytorch_not_available"}
    
    # Format prompt
    prompt = format_prompt(history)
    
    # Try up to 3 times to get a valid equation
    max_retries = 3
    all_outputs = []
    
    for retry in range(max_retries):
        try:
            logger.info("=" * 80)
            logger.info(f"🔄 LLM GENERATION ATTEMPT {retry + 1}/{max_retries}")
            logger.info("=" * 80)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a Nerdle game solver. Output ONLY a valid 8-character equation that balances (e.g., 48-32=16). The equation MUST evaluate to be true (left side equals right side). No explanations, no extra text, just the equation.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # Generate with varying temperature for retries
            temperature = 0.7 + (retry * 0.2)  # Increase temperature for retries
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_outputs.append(generated_text)
            
            # Log the full model output
            logger.info(f"📝 LLM RAW OUTPUT (Attempt {retry + 1}):")
            logger.info(f"   Prompt length: {len(text)} chars")
            logger.info(f"   Generated length: {len(generated_text)} chars")
            logger.info(f"   Full generated text:\n{generated_text}")
            logger.info("-" * 80)
            
            # Extract guess (last line or equation pattern)
            guess = extract_guess_from_response(generated_text, text)
            
            logger.info(f"🔍 Extracted guess: {guess}")
            logger.info(f"   Length: {len(guess)}")
            logger.info(f"   Valid equation: {is_valid_equation(guess)}")
            
            # Validate the guess before returning
            if is_valid_equation(guess):
                logger.info("=" * 80)
                logger.info(f"✅ SUCCESS - Valid LLM-generated equation on attempt {retry + 1}: {guess}")
                logger.info("=" * 80)
                return guess, "llm", {
                    "attempt": retry + 1,
                    "max_attempts": max_retries,
                    "raw_outputs": all_outputs,
                    "temperature": temperature,
                }
            else:
                logger.warning(f"❌ Invalid equation on attempt {retry + 1}: {guess}. Retrying...")
                if retry < max_retries - 1:
                    continue
                else:
                    logger.warning("=" * 80)
                    logger.warning(f"⚠️  ALL {max_retries} LLM ATTEMPTS FAILED - Using fallback")
                    logger.warning("=" * 80)
                    fallback_guess = generate_valid_fallback(history)
                    return fallback_guess, "fallback", {
                        "reason": "all_llm_attempts_failed",
                        "attempts": max_retries,
                        "raw_outputs": all_outputs,
                    }
        
        except Exception as e:
            logger.error(f"❌ ERROR on attempt {retry + 1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if retry == max_retries - 1:
                logger.error("=" * 80)
                logger.error("⚠️  ERROR AFTER ALL RETRIES - Using fallback")
                logger.error("=" * 80)
                fallback_guess = generate_valid_fallback(history)
                return fallback_guess, "fallback", {
                    "reason": "exception_after_retries",
                    "error": str(e),
                    "attempts": max_retries,
                }
    
    # Should not reach here, but just in case
    logger.warning("⚠️  UNEXPECTED FALLBACK")
    fallback_guess = generate_valid_fallback(history)
    return fallback_guess, "fallback", {"reason": "unexpected_path"}


def format_prompt(history: List[Dict[str, Any]]) -> str:
    """Format game history into a prompt for the model."""
    if len(history) == 0:
        return "This is a nerdle game. Make your first guess of an 8-character equation (e.g., 48-32=16)."
    
    lines = [
        "This is a nerdle game.",
        "Legend: g = green (correct spot), p = purple (wrong spot), b = black (not in equation).",
    ]
    
    for idx, item in enumerate(history, 1):
        guess = item.get("guess", "")
        feedback = item.get("feedback", [])
        
        feedback_str = " ".join(
            f"{guess[i]}{'g' if f == 'correct' else 'p' if f == 'present' else 'b'}"
            for i, f in enumerate(feedback)
        )
        lines.append(f"Attempt {idx}: {guess} | Feedback: {feedback_str}")
    
    lines.append("Find the correct Nerdle equation. Output only the next 8-character guess, nothing else.")
    return "\n".join(lines)


def is_valid_equation(guess: str) -> bool:
    """Validate that an equation is valid and balances."""
    import re
    
    # Check length
    if len(guess) != 8:
        return False
    
    # Check allowed characters
    if not all(c in "0123456789+-*/=" for c in guess):
        return False
    
    # Check exactly one equals sign
    if guess.count("=") != 1:
        return False
    
    # Split into left and right
    parts = guess.split("=")
    if len(parts) != 2:
        return False
    
    left, right = parts
    if not left or not right:
        return False
    
    # Check format (no consecutive operators)
    def is_expression_valid(expr: str) -> bool:
        if not re.match(r'^[0-9+\-*/]+$', expr):
            return False
        if not expr[0].isdigit() or not expr[-1].isdigit():
            return False
        prev_op = False
        for char in expr:
            is_op = char in "+-*/"
            if is_op and prev_op:
                return False
            prev_op = is_op
        return True
    
    if not is_expression_valid(left) or not is_expression_valid(right):
        return False
    
    # Evaluate both sides
    try:
        # Safe evaluation - handle operator precedence
        left_val = eval(left)
        right_val = eval(right)
        
        # Check if they're integers and equal
        if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
            if abs(left_val - right_val) < 0.001:  # Allow for floating point errors
                return True
    except:
        pass
    
    return False


def extract_guess_from_response(response: str, original_prompt: str) -> str:
    """Extract the 8-character equation from model response."""
    import re
    
    logger.info("Extracting guess from response...")
    logger.info(f"Response (first 200 chars): {response[:200]}")
    
    # Remove the prompt from response
    response_clean = response
    if original_prompt in response:
        response_clean = response.split(original_prompt)[-1]
        logger.info(f"Removed prompt, cleaned response length: {len(response_clean)}")
    
    # Look for 8-character equation pattern
    pattern = r"[0-9+\-*/=]{8}"
    matches = re.findall(pattern, response_clean)
    logger.info(f"Found {len(matches)} 8-character matches: {matches}")
    
    # Check each match for validity
    for match in reversed(matches):  # Check from last to first
        if len(match) == 8 and is_valid_equation(match):
            logger.info(f"Using valid 8-char match: {match}")
            return match
    
    # Fallback: try to find any equation-like pattern
    pattern = r"[0-9]+\s*[+\-*/]\s*[0-9]+\s*=\s*[0-9]+"
    matches = re.findall(pattern, response_clean.replace(" ", ""))
    logger.info(f"Found {len(matches)} equation pattern matches: {matches}")
    
    for match in reversed(matches):
        if len(match) == 8 and is_valid_equation(match):
            logger.info(f"Using valid equation pattern match: {match}")
            return match
    
    # Last resort: return default valid equation
    logger.warning(f"Could not extract valid guess, using fallback. Response was: {response_clean[:100]}")
    return "48-32=16"


def generate_valid_fallback(history: List[Dict[str, Any]]) -> str:
    """Generate a valid fallback guess if model fails."""
    logger.info("🔧 FALLBACK FUNCTION CALLED")
    
    # Valid equations that balance
    valid_fallbacks = [
        "48-32=16",
        "10+20=30",
        "25+25=50",
        "12*3=36",
        "8*8=64",
        "15+15=30",
        "6+4-2=8",
        "9*2-3=15",
        "7+7*2=21",
        "5*5+1=26",
    ]
    
    # Try to avoid repeating previous guesses
    previous_guesses = [item.get("guess", "") for item in history]
    
    for fallback in valid_fallbacks:
        if fallback not in previous_guesses:
            logger.info(f"✅ Using fallback guess: {fallback}")
            return fallback
    
    # If all fallbacks were used, return the first one
    logger.info(f"✅ Using default fallback: {valid_fallbacks[0]}")
    return valid_fallbacks[0]


def fallback_guess(history: List[Dict[str, Any]]) -> str:
    """Generate a fallback guess if model fails (legacy function for compatibility)."""
    return generate_valid_fallback(history)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
    })


@app.route("/guess", methods=["POST"])
def guess():
    """Generate next guess endpoint."""
    try:
        data = request.json
        history = data.get("history", [])
        attempt_number = data.get("attemptNumber", len(history) + 1)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"GUESS REQUEST - Attempt {attempt_number}")
        logger.info(f"History length: {len(history)}")
        if history:
            logger.info(f"Last guess: {history[-1].get('guess', 'N/A')}")
            logger.info(f"Last feedback: {history[-1].get('feedback', [])}")
        logger.info(f"{'='*80}\n")
        
        # Generate guess (returns tuple: guess, source, metadata)
        guess_result, source, metadata = generate_guess(history, attempt_number)
        
        logger.info("=" * 80)
        logger.info(f"📤 RETURNING RESPONSE:")
        logger.info(f"   Guess: {guess_result}")
        logger.info(f"   Source: {source.upper()}")
        logger.info(f"   Metadata: {metadata}")
        logger.info("=" * 80)
        
        return jsonify({
            "guess": guess_result,
            "source": source,  # "llm" or "fallback"
            "metadata": metadata,
        })
        
    except Exception as e:
        logger.error(f"Error in /guess endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        fallback_guess = generate_valid_fallback(data.get("history", []))
        return jsonify({
            "error": str(e),
            "guess": fallback_guess,
            "source": "fallback",
            "metadata": {"reason": "exception_in_endpoint", "error": str(e)},
        }), 500


@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """Load model endpoint."""
    try:
        data = request.json
        model_name = data.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
        
        load_model(model_name)
        
        return jsonify({
            "status": "success",
            "message": f"Model {model_name} loaded successfully",
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
        }), 500


if __name__ == "__main__":
    # Load model on startup
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    
    try:
        load_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.info("Server will start but model inference will use fallback")
    
    # Run server
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

