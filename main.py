from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, List, Dict
import os
import httpx
from dotenv import load_dotenv
from math import gcd
from functools import reduce

load_dotenv()

EMAIL = os.getenv("OFFICIAL_EMAIL")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)

class BFHLRequest(BaseModel):
    __root__: Dict[str, Union[int, List[int], str]]

@app.get("/health")
def health():
    return {"is_success": True, "official_email": EMAIL}

@app.post("/bfhl")
async def bfhl(req: BFHLRequest):
    body = req.__root__
    if len(body) != 1:
        raise HTTPException(status_code=400, detail={"is_success": False})

    key, value = next(iter(body.items()))

    try:
        if key == "fibonacci":
            data = [0, 1]
            for _ in range(value - 2):
                data.append(data[-1] + data[-2])
            data = data[:value]

        elif key == "prime":
            data = [x for x in value if isinstance(x, int) and is_prime(x)]

        elif key == "lcm":
            data = reduce(lcm, value)

        elif key == "hcf":
            data = reduce(gcd, value)

        elif key == "AI":
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_KEY}",
                    json={"contents":[{"parts":[{"text":value}]}]}
                )
            text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            data = text.split()[-1].strip(".,!?")

        else:
            raise ValueError("Invalid key")

        return {"is_success": True, "official_email": EMAIL, "data": data}

    except Exception:
        raise HTTPException(status_code=400, detail={"is_success": False})