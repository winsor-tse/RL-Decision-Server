from pydantic import BaseModel
from typing import List

class Entity(BaseModel):
    id: int
    mapX: int
    mapY: int
    type: str
    distance: int

class Player(BaseModel):
    id: int
    mapX: int
    mapY: int
    direction: str
    hp: int
    maxHp: int
    mp: int
    maxMp: int

class WorldState(BaseModel):
    timestamp: int
    player: Player
    entities: List[Entity]

"""
#Use Fast API

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# Allow browser access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/ai-action")
async def ai_action(state: WorldState):
    print("Received state:", state)

    action = "left"

    return {"move": action}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""