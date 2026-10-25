import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import random
import time
from collections import deque
import os
import sqlite3
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Neuro-Symbolic Navigation", layout="wide")

# ==========================================
# 0. DATABASE INITIALIZATION
# ==========================================
def init_db():
    conn = sqlite3.connect("simulation_history.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            grid_seed INTEGER,
            outcome TEXT,
            steps_taken INTEGER,
            final_health INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_run(model_name, seed, outcome, steps, health):
    conn = sqlite3.connect("simulation_history.db")
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO runs (timestamp, model_name, grid_seed, outcome, steps_taken, final_health)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, model_name, seed, outcome, steps, health))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect("simulation_history.db")
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC", conn)
    conn.close()
    return df

# Initialize DB when the app loads
init_db()

# ==========================================
# 1. MODEL ARCHITECTURES
# ==========================================
class TargetHunterDQN(nn.Module):
    def __init__(self, input_size=9, output_actions=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, output_actions)
        )
    def forward(self, x): return self.network(x)

class TargetHunterDRQN_LSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=32, output_actions=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_actions)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TargetHunterDRQN_GRU(nn.Module):
    def __init__(self, input_size=9, hidden_size=32, output_actions=4):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_actions)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])

# ==========================================
# 2. CACHED MODEL LOADER
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    paths = {
        "DQN (Single Frame)": ("models/neuro_symbolic_dqn_weights.pt", TargetHunterDQN()),
        "DRQN-LSTM (5-Frame Seq)": ("models/neuro_symbolic_drqn_lstm_weights_11000_episodes.pt", TargetHunterDRQN_LSTM()),
        "DRQN-GRU (5-Frame Seq)": ("models/neuro_symbolic_drqn_gru_weights_5000_episodes.pt", TargetHunterDRQN_GRU())
    }
    for name, (path, arch) in paths.items():
        if os.path.exists(path):
            try:
                arch.load_state_dict(torch.load(path, weights_only=True))
                arch.eval()
                models[name] = arch
            except Exception as e:
                st.sidebar.error(f"Error loading {name}: {e}")
        else:
            st.sidebar.warning(f"Missing file: {path}")
    return models

loaded_models = load_models()

# ==========================================
# 3. ENVIRONMENT ENGINE
# ==========================================
def create_dynamic_env(size=10, num_walls=6, num_fire=3, num_acid=1):
    grid = np.zeros((size, size), dtype=int)
    def get_empty_coords():
        while True:
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if grid[y, x] == 0: return x, y

    for _ in range(num_walls): x, y = get_empty_coords(); grid[y, x] = 1
    for _ in range(num_fire): x, y = get_empty_coords(); grid[y, x] = 2
    for _ in range(num_acid): x, y = get_empty_coords(); grid[y, x] = 3

    ax, ay = get_empty_coords()
    tx, ty = get_empty_coords()
    target_velocity = {"dx": random.choice([-1, 1]), "dy": random.choice([-1, 1])}
    return grid, (ax, ay), (tx, ty), target_velocity

def build_frames(agent_pos, target_pos, target_vel, grid, health=100):
    Agent_Frame = {"position": {"x": agent_pos[0], "y": agent_pos[1]}, "health": health}
    Target_Frame = {"position": {"x": target_pos[0], "y": target_pos[1]}, "velocity": target_vel}
    Obstacles_Frames = []
    terrain_map = {1: ("wall", 0), 2: ("fire_pit", 50), 3: ("acid_trap", 100)}
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] in terrain_map:
                obs_type, dmg = terrain_map[grid[y, x]]
                Obstacles_Frames.append({"position": {"x": x, "y": y}, "type": obs_type, "damage_value": dmg})
    return Agent_Frame, Target_Frame, Obstacles_Frames

def extract_raw_features(agent_frame, target_frame, obstacles_frames):
    rel_x = target_frame["position"]["x"] - agent_frame["position"]["x"]
    rel_y = target_frame["position"]["y"] - agent_frame["position"]["y"]
    dx, dy = target_frame["velocity"]["dx"], target_frame["velocity"]["dy"]
    health = agent_frame["health"]
    ax, ay = agent_frame["position"]["x"], agent_frame["position"]["y"]
    adj = {"north": (ax, ay - 1), "south": (ax, ay + 1), "east": (ax + 1, ay), "west": (ax - 1, ay)}
    hazards = [0, 0, 0, 0]
    for obs in obstacles_frames:
        pos = (obs["position"]["x"], obs["position"]["y"])
        if pos == adj["north"]: hazards[0] = obs["damage_value"]
        elif pos == adj["south"]: hazards[1] = obs["damage_value"]
        elif pos == adj["east"]: hazards[2] = obs["damage_value"]
        elif pos == adj["west"]: hazards[3] = obs["damage_value"]
    return np.array([rel_x, rel_y, dx, dy, health] + hazards, dtype=np.float32)

def scale_features(raw_vector, max_x=10, max_y=10):
    scaled = np.zeros_like(raw_vector)
    scaled[0] = raw_vector[0] / max_x
    scaled[1] = raw_vector[1] / max_y
    scaled[2] = raw_vector[2]
    scaled[3] = raw_vector[3]
    scaled[4] = raw_vector[4] / 100.0
    scaled[5:] = raw_vector[5:] / 100.0
    return scaled

def weka_approved_action(proposed_action, agent_pos, grid):
    x, y = agent_pos
    is_illegal = False
    if proposed_action == "move_north" and y > 0 and grid[y-1, x] == 1: is_illegal = True
    elif proposed_action == "move_south" and y < 9 and grid[y+1, x] == 1: is_illegal = True
    elif proposed_action == "move_east" and x < 9 and grid[y, x+1] == 1: is_illegal = True
    elif proposed_action == "move_west" and x > 0 and grid[y, x-1] == 1: is_illegal = True
    
    if proposed_action == "move_north" and y == 0: is_illegal = True
    if proposed_action == "move_south" and y == 9: is_illegal = True
    if proposed_action == "move_west" and x == 0: is_illegal = True
    if proposed_action == "move_east" and x == 9: is_illegal = True
    
    if is_illegal: return "stop"
    return proposed_action

# ==========================================
# 4. STREAMLIT INTERFACE
# ==========================================
st.title("🤖 Neuro-Symbolic Agent Navigation Demo")

st.sidebar.header("Control Panel")
selected_model = st.sidebar.selectbox("Active Brain", list(loaded_models.keys()) if loaded_models else ["None"])
sim_speed = st.sidebar.slider("Step Delay (Seconds)", 0.05, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🎲 Environment Generator")

if "grid_seed" not in st.session_state:
    st.session_state.grid_seed = 42

if st.sidebar.button("🎲 Randomize Grid Layout"):
    st.session_state.grid_seed = random.randint(1, 9999)

seed_val = st.sidebar.number_input("Grid Seed", value=st.session_state.grid_seed, step=1)
num_walls = st.sidebar.slider("Number of Walls", 0, 20, 4)
num_fire = st.sidebar.slider("Number of Fire Pits", 0, 10, 2)
num_acid = st.sidebar.slider("Number of Acid Traps", 0, 5, 1)

start_sim = st.sidebar.button("▶️ Start Navigation", type="primary")

grid_display = st.empty()
metrics_display = st.empty()

if start_sim and loaded_models:
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    grid, agent_pos, target_pos, target_vel = create_dynamic_env(
        size=10, num_walls=num_walls, num_fire=num_fire, num_acid=num_acid
    )
    grid_size = 10
    agent_health = 100
    agent_trail = []
    
    model = loaded_models[selected_model]
    is_dqn = "DQN" in selected_model and "DRQN" not in selected_model
    state_window = deque([np.zeros(9)] * 5, maxlen=5)
    
    actions = ["move_north", "move_south", "move_east", "move_west"]
    done = False
    
    for step in range(1, 76): 
        if done: break
        agent_trail.append(list(agent_pos))
        
        # 1. Feature extraction
        Agent_Frame, Target_Frame, Obstacles_Frames = build_frames(agent_pos, target_pos, target_vel, grid, health=agent_health)
        scaled_feats = scale_features(extract_raw_features(Agent_Frame, Target_Frame, Obstacles_Frames))
        
        # 2. Inference
        if is_dqn:
            tensor_in = torch.FloatTensor(scaled_feats).unsqueeze(0)
        else:
            state_window.append(scaled_feats)
            tensor_in = torch.FloatTensor(np.array(state_window)).unsqueeze(0)
            
        with torch.no_grad():
            q_values = model(tensor_in).numpy()[0]
            
        raw_act = actions[int(np.argmax(q_values))]
        
        # 3. WEKA Veto
        final_act = weka_approved_action(raw_act, agent_pos, grid)
        move_act = final_act if final_act != "stop" else random.choice(actions)
        
        # --- LOOP BREAKER ---
        if len(agent_trail) >= 4 and list(agent_pos) == agent_trail[-2] and agent_trail[-1] == agent_trail[-3]:
            valid_moves = [a for a in actions if weka_approved_action(a, agent_pos, grid) != "stop"]
            if valid_moves:
                move_act = random.choice(valid_moves)

        # 4. Agent Movement
        ax, ay = agent_pos
        if move_act == "move_north": ay -= 1
        elif move_act == "move_south": ay += 1
        elif move_act == "move_east": ax += 1
        elif move_act == "move_west": ax -= 1
        
        ax, ay = max(0, min(9, ax)), max(0, min(9, ay))
        
        if grid[ay, ax] != 1:  
            agent_pos = (ax, ay)
            if grid[ay, ax] == 2: agent_health -= 50
            elif grid[ay, ax] == 3: agent_health = 0; done = True
            
        # --- TARGET BOUNCE PHYSICS ---
        tx, ty = target_pos
        dx, dy = target_vel["dx"], target_vel["dy"]
        
        if tx + dx < 0 or tx + dx > 9: dx *= -1
        if ty + dy < 0 or ty + dy > 9: dy *= -1
        
        target_pos = (tx + dx, ty + dy)
        target_vel["dx"] = dx
        target_vel["dy"] = dy
        
        if agent_pos == target_pos: done = True
        
        # 5. Render
        html = "<table style='border-collapse: collapse; margin: auto; background-color: #0e1117;'>"
        for y in range(grid_size):
            html += "<tr>"
            for x in range(grid_size):
                cell_style = "width: 48px; height: 48px; border: 1px solid #333; text-align: center; font-size: 22px;"
                cell_val = grid[y, x]
                
                if (x, y) == agent_pos:
                    html += f"<td style='{cell_style} background-color: #2e86c1;'>🤖</td>"
                elif (x, y) == target_pos:
                    html += f"<td style='{cell_style} background-color: #f1c40f;'>🎯</td>"
                elif cell_val == 1:
                    html += f"<td style='{cell_style} background-color: #7f8c8d;'>🧱</td>"
                elif cell_val == 2:
                    html += f"<td style='{cell_style} background-color: #e67e22;'>🔥</td>"
                elif cell_val == 3:
                    html += f"<td style='{cell_style} background-color: #27ae60;'>🧪</td>"
                elif [x, y] in agent_trail:
                    html += f"<td style='{cell_style} background-color: #1a252f; color: #555;'>•</td>"
                else:
                    html += f"<td style='{cell_style}'></td>"
            html += "</tr>"
        html += "</table>"
        
        grid_display.markdown(html, unsafe_allow_html=True)
        
        with metrics_display.container():
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Step / Health", f"{step}/75 | HP: {agent_health}")
            c2.metric("Target Velocity", f"dX:{target_vel['dx']} dY:{target_vel['dy']}")
            c3.metric("Neural Decision", raw_act.replace('move_', '').upper())
            c4.metric("WEKA Veto Filter", " Approved" if final_act != "stop" else "VETOED")
            
        time.sleep(sim_speed)
        
    # --- LOGGING TO SQLITE ---
    if agent_pos == target_pos:
        outcome = "Intercepted Target"
        st.success(f" {outcome} in {step} steps!")
        st.balloons()
    elif agent_health <= 0:
        outcome = "Died (Acid Trap)"
        st.error(f"{outcome} at step {step}")
    else:
        outcome = "Target Evaded (Max Steps)"
        st.warning(f" {outcome}")

    # Write results to database
    log_run(selected_model, seed_val, outcome, step, agent_health)

# ==========================================
# 5. DATABASE HISTORY VIEWER
# ==========================================
st.markdown("---")
st.subheader("📊 SQLite Simulation History")
history_df = load_history()

if not history_df.empty:
    st.dataframe(
        history_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "id": "Run ID",
            "timestamp": "Time",
            "model_name": "AI Model",
            "grid_seed": "Grid Seed",
            "outcome": "Outcome",
            "steps_taken": "Steps",
            "final_health": "End Health"
        }
    )
else:
    st.info("No runs recorded yet. Run a simulation to log data to the SQLite database.")