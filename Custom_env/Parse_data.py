from typing import List

DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "idle": 4  # if needed
}

def parse_observation(data: dict) -> List[float]:
    obs = []
    player = data["player"]
    direction = DIRECTION_MAP.get(player.get("direction", "idle"), 4)
    obs.extend([
        player["mapX"],
        player["mapY"],
        direction,
        player["hp"],
        player["maxHp"],
        player["mp"],
        player["maxMp"]
    ])
    monsters = [
        e for e in data["entities"]
        if e["type"] == "monster" and not e["isCurrentPlayer"]
    ]
    monsters.sort(key=lambda m: m.get("distance", float('inf')))

    for monster in monsters[:2]:
        obs.extend([
            monster["mapX"],
            monster["mapY"],
            0, 
            monster["hp"],
            monster["maxHp"],
            monster["mp"],
            monster["maxMp"]
        ])

    while len(obs) < 21:
        obs.extend([0.0] * 7)
    return obs

def get_reward(obs, actions):
    #simple implementation of rewards
    #Implement based on the given 21 OBS Space
    #Attacking give positive rewards
    #Distance from Enemy over a certain limit will give positive rewards
    #Killing (Track this somehow...) enemy will give positive rewards
    #Hp below a certain limit is negative rewards, etc.
    #Rest can be zero reward...
    return None