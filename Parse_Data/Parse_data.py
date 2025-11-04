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
