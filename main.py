from fastapi import FastAPI, HTTPException
from pydantic import RootModel
from typing import Union, List, Dict
import os
import httpx
from dotenv import load_dotenv
from math import gcd
from functools import reduce

# Load environment variables
load_dotenv()

EMAIL = os.getenv("OFFICIAL_EMAIL", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(title="BFHL API", version="1.0.0")


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def lcm(a: int, b: int) -> int:
    # Prevent overflow on weird inputs
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


# ---- Pydantic v2 RootModel for arbitrary top-level dict ----
class BFHLRequest(RootModel[Dict[str, Union[int, List[int], str]]]):
    pass


@app.get("/health")
def health():
    """
    Simple health endpoint to verify service and env wiring.
    """
    return {
        "is_success": True,
        "official_email": EMAIL,
    }


@app.post("/bfhl")
async def bfhl(req: BFHLRequest):
    """
    Accepts a single-key JSON object with one of the following keys:
      - {"fibonacci": <n:int>}                  -> returns first n Fibonacci numbers
      - {"prime": [<ints>]}                     -> returns primes from the list
      - {"lcm":   [<ints>]}                     -> returns LCM of numbers
      - {"hcf":   [<ints>]}                     -> returns GCD/HCF of numbers
      - {"AI":    "<prompt:str>"}               -> calls Gemini and returns last token (per your original logic)

    Response:
    {
      "is_success": true,
      "official_email": "<EMAIL>",
      "data": <result>
    }
    """
    body = req.root

    # Ensure single key
    if not isinstance(body, dict) or len(body) != 1:
        raise HTTPException(
            status_code=400,
            detail={"is_success": False, "error": "Payload must contain exactly one key."},
        )

    key, value = next(iter(body.items()))

    try:
        if key == "fibonacci":
            if not isinstance(value, int) or value < 0:
                raise ValueError("`fibonacci` expects a non-negative integer.")
            # Generate first n Fibonacci numbers
            if value == 0:
                data: List[int] = []
            elif value == 1:
                data = [0]
            else:
                data = [0, 1]
                for _ in range(value - 2):
                    data.append(data[-1] + data[-2])

        elif key == "prime":
            if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
                raise ValueError("`prime` expects a list of integers.")
            data = [x for x in value if is_prime(x)]

        elif key == "lcm":
            if not isinstance(value, list) or len(value) == 0 or not all(isinstance(x, int) for x in value):
                raise ValueError("`lcm` expects a non-empty list of integers.")
            data = reduce(lcm, value)

        elif key == "hcf":
            if not isinstance(value, list) or len(value) == 0 or not all(isinstance(x, int) for x in value):
                raise ValueError("`hcf` expects a non-empty list of integers.")
            data = reduce(gcd, value)

        elif key == "AI":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("`AI` expects a non-empty string prompt.")
            if not GEMINI_KEY:
                raise ValueError("GEMINI_API_KEY is not set in environment.")

            # Call Gemini (v1beta generateContent)
            async with httpx.AsyncClient(timeout=30) as client:
                res = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_KEY}",
                    json={"contents": [{"parts": [{"text": value}]}]},
                )
                res.raise_for_status()

            j = res.json()
            # Defensive parsing: structure can vary; keep your original behavior where possible
            try:
                text = j["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                raise ValueError("Unexpected response from Gemini API.")

            # Original requirement: return the last token stripped of punctuation
            data = text.split()[-1].strip(".,!?")

        else:
            raise ValueError("Invalid key. Use one of: fibonacci, prime, lcm, hcf, AI")

        return {
            "is_success": True,
            "official_email": EMAIL,
            "data": data,
        }

    except HTTPException:
        # Already well-formed; re-raise
        raise
    except Exception as e:
        # Helpful error during development/deployment
        raise HTTPException(
            status_code=400,
            detail={"is_success": False, "error": str(e)},
        )
