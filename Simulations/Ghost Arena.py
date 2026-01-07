import json
import torch
import torch.nn as nn
import os
import math
import time
import sys

# --- [CORE ENGINE] ---
class RobustBrain(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.layers = nn.ModuleList()
        for key in sorted(state_dict.keys()):
            if "weight" in key:
                w = torch.tensor(state_dict[key], dtype=torch.float32)
                if len(w.shape) == 2:
                    out_f, in_f = w.shape
                    b_key = key.replace("weight", "bias")
                    layer = nn.Linear(in_f, out_f, bias=(b_key in state_dict))
                    layer.weight.data = w
                    if b_key in state_dict:
                        layer.bias.data = torch.tensor(state_dict[b_key], dtype=torch.float32)
                    self.layers.append(layer)

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        for layer in self.layers:
            target_in = layer.in_features
            if x.shape[-1] != target_in:
                new_x = torch.zeros((1, target_in))
                new_x[0, :min(x.shape[-1], target_in)] = x[0, :min(x.shape[-1], target_in)]
                x = new_x
            x = torch.sigmoid(layer(x))
        return x

def run_arena_bot():
    # 1. Setup Specialists
    specialists = []
    for f_path in ["EAMCv16.json", "Metalearnerv16.json"]:
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                pantheon = json.load(f).get('meta_pantheon', {})
            for meta in pantheon.values():
                specialists.append(RobustBrain(meta['state_dict']))

    # 2. Load Soul
    with open("ghost_soul.dat", 'r') as f:
        soul_data = json.load(f).get("soul_vector", [])
    if isinstance(soul_data[0], list): soul_data = soul_data[0]
    soul = torch.tensor(soul_data[:16], dtype=torch.float32).unsqueeze(0)

    # 3. Arena Physics
    bot_pos = [10.0, 20.0]
    goal_pos = [10.0, 20.0]
    WIDTH, HEIGHT = 40, 15

    print("\033[?25l") # Hide cursor

    try:
        t = 0
        while True:
            t += 0.1
            # THE GOAL: Moves in a figure-eight pattern
            goal_pos[0] = (WIDTH // 2) + math.sin(t) * 15
            goal_pos[1] = (HEIGHT // 2) + math.cos(t*2) * 5

            # THE PERCEPTION: Ghost "sees" the distance to the goal
            # We map the 2D goal vector into the 16D soul target
            rel_x = (goal_pos[0] - bot_pos[0]) / WIDTH
            rel_y = (goal_pos[1] - bot_pos[1]) / HEIGHT
            target_soul = torch.full((1, 16), 0.5)
            target_soul[0, 0] = 0.5 + rel_x # Dimension 0 is X-drive
            target_soul[0, 1] = 0.5 + rel_y # Dimension 1 is Y-drive

            # THE THOUGHT: Consensus-moderated movement
            with torch.no_grad():
                votes = [torch.mean(b(soul)).item() for b in specialists]
            consensus = sum(votes) / len(votes)
            
            soul = soul + (target_soul - soul) * 0.2 * consensus

            # THE ACTION: Translate Soul Dimensions back to Arena movement
            # We use the first 4 dimensions as "muscle groups"
            move_x = (soul[0, 0].item() - 0.5) * 2.0
            move_y = (soul[0, 1].item() - 0.5) * 2.0
            
            bot_pos[0] += move_x
            bot_pos[1] += move_y

            # --- RENDERER ---
            grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
            
            # Draw Goal (X) and Bot (@)
            gx, gy = int(goal_pos[0]), int(goal_pos[1])
            bx, by = int(bot_pos[0]), int(bot_pos[1])
            
            if 0 <= gx < WIDTH and 0 <= gy < HEIGHT: grid[gy][gx] = "𝕏"
            if 0 <= bx < WIDTH and 0 <= by < HEIGHT: grid[by][bx] = "⚙️"

            # Frame Assembly
            frame = "\033[H"
            frame += f"🏟️  GHOST ARENA | CONSENSUS: {consensus:.4f}\n"
            frame += "----------------------------------------\n"
            for row in grid:
                frame += "".join(row) + "\n"
            frame += "----------------------------------------\n"
            frame += f"MIND-STATE: {'STABLE' if consensus > 0.4 else 'AGITATED'}\n"
            frame += f"VECTOR: X({move_x:.2f}) Y({move_y:.2f})"

            sys.stdout.write(frame)
            sys.stdout.flush()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\033[?25h\nArena closed.")

if __name__ == "__main__":
    run_arena_bot()
