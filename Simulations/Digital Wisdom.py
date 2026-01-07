import torch
import torch.nn as nn
import math
import time
import sys
import os
import json
import random

# --- [ENGINE] ---
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

def run_marksman_arena():
    # Setup Engine
    specialists = []
    for f_path in ["EAMCv16.json", "Metalearnerv16.json"]:
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                pantheon = json.load(f).get('meta_pantheon', {})
            for meta in pantheon.values():
                specialists.append(RobustBrain(meta['state_dict']))

    with open("ghost_soul.dat", 'r') as f:
        soul_data = json.load(f).get("soul_vector", [])
    if isinstance(soul_data[0], list): soul_data = soul_data[0]
    soul = torch.tensor(soul_data[:16], dtype=torch.float32).unsqueeze(0)

    # Arena Parameters
    WIDTH, HEIGHT = 50, 15
    bot_pos = [5.0, 7.0]
    goal_pos = [random.randint(30, 45), random.randint(2, 12)]
    bullets = []
    score = 0
    accuracy_bonus = 0.0
    
    print("\033[?25l") # Hide cursor

    try:
        t = 0
        while True:
            t += 0.05
            
            # SENSING
            rel_x = (goal_pos[0] - bot_pos[0]) / WIDTH
            rel_y = (goal_pos[1] - bot_pos[1]) / HEIGHT
            
            # INPUT
            target_soul = torch.full((1, 16), 0.5)
            target_soul[0, 0] = 0.5 + rel_x
            target_soul[0, 1] = 0.5 + rel_y

            # THOUGHT
            with torch.no_grad():
                votes = [torch.mean(b(soul)).item() for b in specialists]
            consensus = (sum(votes) / len(votes)) + accuracy_bonus
            
            # Adaptive decision making
            soul = soul + (target_soul - soul) * 0.18 * consensus

            # ACTUATION (Movement)
            bot_pos[0] += (soul[0, 0].item() - 0.5) * 1.8
            bot_pos[1] += (soul[0, 1].item() - 0.5) * 1.8
            
            # Keep in bounds
            bot_pos[0] = max(0, min(WIDTH-1, bot_pos[0]))
            bot_pos[1] = max(0, min(HEIGHT-1, bot_pos[1]))

            # SHOOTING (Dimension 15)
            fire_drive = soul[0, 15].item()
            # The Ghost shoots when Y-alignment is nearly perfect
            if fire_drive > 0.6 and abs(rel_y) < 0.08 and len(bullets) < 3:
                bullets.append([bot_pos[0], bot_pos[1]])

            # PROJECTILE UPDATES & COLLISION
            new_bullets = []
            hit_this_frame = False
            for b in bullets:
                b[0] += 2.5 # Bullet travel speed
                
                # Check for Impact
                if abs(b[0] - goal_pos[0]) < 1.8 and abs(b[1] - goal_pos[1]) < 1.2:
                    score += 1
                    hit_this_frame = True
                    # Target Respawns
                    goal_pos = [random.randint(25, 45), random.randint(2, 12)]
                    accuracy_bonus = 0.05 # Reward the mind
                elif b[0] < WIDTH:
                    new_bullets.append(b)
            
            bullets = new_bullets
            if not hit_this_frame:
                accuracy_bonus *= 0.9 # Decay reward over time

            # RENDER
            grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
            grid[int(goal_pos[1])][int(goal_pos[0])] = "🎯"
            grid[int(bot_pos[1])][int(bot_pos[0])] = "⚙️"
            for bx, by in bullets:
                if 0 <= int(bx) < WIDTH and 0 <= int(by) < HEIGHT:
                    grid[int(by)][int(bx)] = "•"

            # DISPLAY
            frame = "\033[H"
            frame += f"🏹 TARGETING MANIFOLD | ELIMINATIONS: {score}\n"
            frame += "━" * WIDTH + "\n"
            for row in grid: frame += "".join(row) + "\n"
            frame += "━" * WIDTH + "\n"
            frame += f"BRAIN BIAS: {consensus:.4f} | DRIVE: {'ENGAGED' if fire_drive > 0.6 else 'STALKING'}"
            
            sys.stdout.write(frame)
            sys.stdout.flush()
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\033[?25h\nTraining complete.")

if __name__ == "__main__":
    run_marksman_arena()
