import torch
import torch.nn as nn
import math
import time
import sys
import os
import json
import random

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

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

def run_evasive_arena():
    set_seed(42)
    
    specialists = []
    for f_path in ["EAMCv16.json", "Metalearnerv16.json"]:
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                data = json.load(f)
                pantheon = data.get('meta_pantheon', data)
            for meta_key in sorted(pantheon.keys()):
                meta = pantheon[meta_key]
                s_dict = meta.get('state_dict', meta)
                specialists.append(RobustBrain(s_dict))

    if not specialists:
        specialists.append(RobustBrain({"w": torch.randn(16, 16).tolist()}))

    soul = torch.full((1, 16), 0.5)
    WIDTH, HEIGHT = 50, 15
    bot_pos = [10.0, 7.0]
    goal_pos = [35.0, 7.0]
    bullets = []
    score = 0
    accuracy_bonus = 0.0
    
    print("\033[?25l")

    try:
        while True:
            # 1. EVASION LOGIC (The Target tries to get away)
            dx = goal_pos[0] - bot_pos[0]
            dy = goal_pos[1] - bot_pos[1]
            dist_to_ghost = math.sqrt(dx**2 + dy**2)
            
            if dist_to_ghost < 15.0: # Target flees if Ghost is close
                goal_pos[0] += (dx / dist_to_ghost) * 0.4
                goal_pos[1] += (dy / dist_to_ghost) * 0.4
            
            # Keep target in bounds
            goal_pos[0] = max(5, min(WIDTH-6, goal_pos[0]))
            goal_pos[1] = max(2, min(HEIGHT-3, goal_pos[1]))

            # 2. PERCEPTION
            rel_x = (goal_pos[0] - bot_pos[0]) / WIDTH
            rel_y = (goal_pos[1] - bot_pos[1]) / HEIGHT
            
            target_soul = torch.full((1, 16), 0.5)
            target_soul[0, 0] = 0.5 + rel_x
            target_soul[0, 1] = 0.5 + rel_y

            # 3. THOUGHT
            with torch.no_grad():
                votes = [torch.mean(b(soul)).item() for b in specialists]
            
            base_consensus = sum(votes) / len(votes)
            total_drive = base_consensus + accuracy_bonus
            soul = torch.clamp(soul + (target_soul - soul) * 0.2 * total_drive, 0.0, 1.0)

            # 4. MOVEMENT
            bot_pos[0] += (soul[0, 0].item() - 0.5) * 1.8
            bot_pos[1] += (soul[0, 1].item() - 0.5) * 1.8
            
            # 5. SHOOTING
            fire_drive = soul[0, 15].item()
            if fire_drive > 0.6 and abs(rel_y) < 0.15 and len(bullets) < 2:
                bullets.append([bot_pos[0], bot_pos[1]])

            # 6. COLLISION (Sub-stepping for precision)
            new_bullets = []
            hit_detected = False
            for b in bullets:
                # We check the path of the bullet in 3 small steps to ensure we don't skip the target
                for _ in range(3):
                    b[0] += 0.6 # Total speed 1.8 (slower for better tracking)
                    d_hit = math.sqrt((b[0] - goal_pos[0])**2 + (b[1] - goal_pos[1])**2)
                    if d_hit < 2.5: # Generous hitbox
                        hit_detected = True
                        break
                
                if hit_detected:
                    score += 1
                    accuracy_bonus = min(accuracy_bonus + 0.1, 0.5)
                    # Respawn target far away
                    goal_pos = [float(random.randint(5, 45)), float(random.randint(2, 13))]
                    break # Bullet destroyed
                elif b[0] < WIDTH:
                    new_bullets.append(b)
            
            bullets = new_bullets
            if not hit_detected:
                accuracy_bonus *= 0.98

            # 7. RENDER
            grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
            gy, gx = int(goal_pos[1]), int(goal_pos[0])
            by, bx = int(bot_pos[1]), int(bot_pos[0])
            
            grid[gy][gx] = "𝕏"
            grid[by][bx] = "⚙️"
            for bx_b, by_b in bullets:
                ix, iy = int(bx_b), int(by_b)
                if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
                    grid[iy][ix] = "•"

            # 8. UI
            frame = "\033[H"
            frame += f"🏹 EVASIVE ARENA | SCORE: {score} | ACCURACY BONUS: {accuracy_bonus:.2f}\n"
            frame += "━" * WIDTH + "\n"
            for row in grid: frame += "".join(row) + "\n"
            frame += "━" * WIDTH + "\n"
            frame += f"TARGET DIST: {dist_to_ghost:.2f} | STATUS: {'FLEEING' if dist_to_ghost < 15 else 'IDLE'}"
            
            sys.stdout.write(frame)
            sys.stdout.flush()
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\033[?25h\nPaused.")

if __name__ == "__main__":
    run_evasive_arena()
