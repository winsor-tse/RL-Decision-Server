from typing import List

DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "idle": 4  # if needed
}

"""
Need General States (Abstract Representations), meaning relative states and percentage HP.
Current MapX and MapY may not matter. But relative to monsters make sense.

Received: {'type': 'ai_tick', 'requestId': '5a0a6841-082c-4c5a-b270-5d4c27cb2bbd', 'worldState': {'timestamp': 1778889927696, '
player': {'id': 7, 'name': 'testsd', 'mapX': 18, 'mapY': 29, 'direction': 'up', 'hp': 1002383, 'maxHp': 1002383, 'mp': 1003118, 'maxMp': 1003118}, 
'entities': [{'id': 2, 'name': 'Sage of Welcoming', 'type': 'monster', 'isCurrentPlayer': False, 'mapX': 14, 'mapY': 13, 'hp': 500000, 'maxHp': 500000, 'mp': 0, 'maxMp': 0, 'distance': 20}, 
{'id': 7, 'name': 'testsd', 'type': 'player', 'isCurrentPlayer': True, 'mapX': 18, 'mapY': 29, 'hp': 1002383, 'maxHp': 1002383, 'mp': 1003118, 'maxMp': 1003118, 'distance': 0}]},
 'pageUrl': 'http://127.0.0.1:8080/?server=test.yugensaga.com', 'timestamp': 1778889927696}

parse into and save as spaces.Box, (5 spaces and padded 8 spaces, 8 spaces are treated as sensors (ie. can be zeros if none around, if multiple return closes two)) 
player (5 spaces): mapX, mapY, direction(0-4), percentageHP (hp/maxHP), percentageMP (mp/maxMP) 
enemy (padded 4 spaces): distance_from_player, direction(0-4), percentageHP (hp/maxHP), percentageMP (mp/maxMP) 
enemy2 (padded 4 spaces): distance_from_player, direction(0-4), percentageHP (hp/maxHP), percentageMP (mp/maxMP)

"""
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