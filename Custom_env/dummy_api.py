# dummy_api.py
from fastapi import FastAPI, Request
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from 127.0.0.1:8080 (JS client)
origins = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",  # include both to be safe
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_action = {}
last_observation = {}

@app.post("/observation")
async def post_observation(request: Request):
    global last_observation
    last_observation = await request.json()
    return {"status": "observation received"}

@app.get("/observation")
def get_observation():
    return last_observation or {"obs": [0]*21} 

@app.post("/last-action")
async def post_action(request: Request):
    global last_action
    last_action = await request.json()
    return {"status": "action received"}

@app.get("/last-action")
def get_action():
    return last_action or {}

@app.post("/reset")
async def reset():
    global last_observation, last_action
    last_observation = {}
    last_action = {}
    return {"status": "reset done"}

def run():
    uvicorn.run(app, host="127.0.0.1", port=6060, log_level="warning")
