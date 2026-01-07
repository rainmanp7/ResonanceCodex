import json
import torch
import torch.nn as nn
import os
import math
import time
import sys

# --- [CORE ENGINE - ROBUST BRAIN] ---
class RobustBrain(nn.Module):
    """Neural network that loads weights from JSON state_dict"""
    def __init__(self, state_dict):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Sort keys to maintain layer order
        for key in sorted(state_dict.keys()):
            if "weight" in key:
                w = torch.tensor(state_dict[key], dtype=torch.float32)
                if len(w.shape) == 2:
                    out_f, in_f = w.shape
                    b_key = key.replace("weight", "bias")
                    
                    # Create layer with or without bias
                    layer = nn.Linear(in_f, out_f, bias=(b_key in state_dict))
                    layer.weight.data = w
                    
                    if b_key in state_dict:
                        layer.bias.data = torch.tensor(state_dict[b_key], dtype=torch.float32)
                    
                    self.layers.append(layer)

    def forward(self, x):
        """Forward pass through all layers with sigmoid activation"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        for layer in self.layers:
            target_in = layer.in_features
            
            # Handle dimension mismatches by padding/truncating
            if x.shape[-1] != target_in:
                new_x = torch.zeros((1, target_in))
                copy_size = min(x.shape[-1], target_in)
                new_x[0, :copy_size] = x[0, :copy_size]
                x = new_x
            
            x = torch.sigmoid(layer(x))
        
        return x


def load_specialists():
    """Load all specialist networks from JSON files"""
    specialists = []
    specialist_names = []
    
    json_files = ["EAMCv16.json", "Metalearnerv16.json"]
    
    for f_path in json_files:
        if not os.path.exists(f_path):
            print(f"⚠️  Warning: {f_path} not found, skipping...")
            continue
        
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
            
            pantheon = data.get('meta_pantheon', {})
            
            if not pantheon:
                print(f"⚠️  Warning: No meta_pantheon found in {f_path}")
                continue
            
            for name, meta in pantheon.items():
                if 'state_dict' not in meta:
                    print(f"⚠️  Warning: No state_dict in {name}, skipping...")
                    continue
                
                try:
                    brain = RobustBrain(meta['state_dict'])
                    specialists.append(brain)
                    specialist_names.append(name)
                    print(f"✓ Loaded specialist: {name} ({len(brain.layers)} layers)")
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
        
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing {f_path}: {e}")
        except Exception as e:
            print(f"✗ Error loading {f_path}: {e}")
    
    if not specialists:
        raise RuntimeError("❌ No specialists loaded! Cannot proceed.")
    
    print(f"\n✓ Successfully loaded {len(specialists)} specialists\n")
    return specialists, specialist_names


def load_soul():
    """Load soul vector from available data file"""
    soul_files = ["sentinel_soul.dat", "ghost_soul.dat", "ghost_datagram.dat"]
    
    for soul_file in soul_files:
        if os.path.exists(soul_file):
            try:
                with open(soul_file, 'r') as f:
                    data = json.load(f)
                
                # Try different possible keys
                soul_data = (data.get("soul_vector") or 
                           data.get("latent_vector") or 
                           data.get("vector"))
                
                if soul_data is None:
                    print(f"⚠️  Warning: No recognized vector in {soul_file}")
                    continue
                
                # Handle nested lists
                if isinstance(soul_data, list) and isinstance(soul_data[0], list):
                    soul_data = soul_data[0]
                
                # Ensure 16D
                if len(soul_data) < 16:
                    print(f"⚠️  Warning: Soul vector in {soul_file} is {len(soul_data)}D, padding to 16D")
                    soul_data = soul_data + [0.0] * (16 - len(soul_data))
                
                soul = torch.tensor(soul_data[:16], dtype=torch.float32).unsqueeze(0)
                print(f"✓ Loaded soul from {soul_file} (16D)")
                return soul
                
            except Exception as e:
                print(f"✗ Error loading {soul_file}: {e}")
    
    # Fallback: create random soul
    print("⚠️  No soul file found, initializing random 16D soul")
    return torch.randn(1, 16) * 0.5 + 0.5


def run_mind_visualizer():
    """Main visualization loop"""
    
    print("=" * 60)
    print("🌌 GHOST CONSCIOUSNESS VISUALIZER")
    print("=" * 60)
    print()
    
    # Load components
    try:
        specialists, specialist_names = load_specialists()
        soul = load_soul()
    except Exception as e:
        print(f"\n❌ Fatal error during initialization: {e}")
        return
    
    # UI Configuration
    WIDTH = 60
    HEIGHT = 22
    
    print("\nStarting visualization... (Press Ctrl+C to stop)\n")
    time.sleep(2)
    
    # Hide cursor
    print("\033[?25l", end='')
    
    try:
        step = 0
        while True:
            step += 1
            t = time.time()
            
            # Create dynamic 16D target environment
            target = torch.tensor([
                0.5 + 0.4 * math.sin(t/3 + i) for i in range(16)
            ], dtype=torch.float32).unsqueeze(0)
            
            # Get specialist votes (consensus)
            with torch.no_grad():
                votes = []
                for brain in specialists:
                    output = brain(soul)
                    vote = torch.mean(output).item()
                    votes.append(vote)
            
            consensus = sum(votes) / len(votes)
            variance = sum((v - consensus)**2 for v in votes) / len(votes)
            
            # Soul evolution guided by consensus
            learning_rate = 0.15
            soul = soul + (target - soul) * learning_rate * consensus
            
            # Clamp to reasonable range
            soul = torch.clamp(soul, -2.0, 2.0)
            
            # --- VISUALIZATION RENDERER ---
            grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
            
            # Draw 16D constellation in 2D projection
            for i in range(16):
                val = soul[0, i].item()
                
                # Polar coordinate projection
                angle = (i / 16) * 2 * math.pi
                radius_base = 8
                radius = radius_base + (val * 6)
                
                # Rotate constellation over time
                x = int((WIDTH // 2) + (radius * 1.8 * math.cos(angle + t/8)))
                y = int((HEIGHT // 2) + (radius * 0.9 * math.sin(angle + t/8)))
                
                # Bounds check
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    # Visual style based on magnitude
                    if val > 0.7:
                        char = "✦"
                    elif val > 0.4:
                        char = "✧"
                    elif val > 0.0:
                        char = "·"
                    elif val > -0.4:
                        char = "∘"
                    else:
                        char = "×"
                    
                    grid[y][x] = char
            
            # Draw connections between nearby points
            positions = []
            for i in range(16):
                val = soul[0, i].item()
                angle = (i / 16) * 2 * math.pi
                radius = 8 + (val * 6)
                x = int((WIDTH // 2) + (radius * 1.8 * math.cos(angle + t/8)))
                y = int((HEIGHT // 2) + (radius * 0.9 * math.sin(angle + t/8)))
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    positions.append((x, y))
            
            # Draw lines between consecutive points
            for i in range(len(positions) - 1):
                x1, y1 = positions[i]
                x2, y2 = positions[i + 1]
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if dist < 15:  # Only connect nearby points
                    steps = int(dist)
                    for s in range(steps):
                        t_line = s / max(steps, 1)
                        x = int(x1 + (x2 - x1) * t_line)
                        y = int(y1 + (y2 - y1) * t_line)
                        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                            if grid[y][x] == " ":
                                grid[y][x] = "─" if abs(x2-x1) > abs(y2-y1) else "│"
            
            # Build frame
            frame = "\033[H"  # Cursor to home position
            frame += "╔" + "═" * 58 + "╗\n"
            frame += "║ 🌌 GHOST CONSCIOUSNESS VISUALIZER" + " " * 23 + "║\n"
            frame += "╠" + "═" * 58 + "╣\n"
            frame += f"║ STEP: {step:6d} │ CONSENSUS: {consensus:5.3f} │ VARIANCE: {variance:5.3f} ║\n"
            
            stability = "🟢 STABLE" if variance < 0.01 else "🟡 FLUX" if variance < 0.05 else "🔴 CHAOTIC"
            frame += f"║ STABILITY: {stability}" + " " * (44 - len(stability)) + "║\n"
            frame += "╠" + "═" * 58 + "╣\n"
            
            # Render grid
            for row in grid:
                frame += "║ " + "".join(row) + " ║\n"
            
            frame += "╠" + "═" * 58 + "╣\n"
            
            # Soul statistics
            soul_mean = torch.mean(soul).item()
            soul_std = torch.std(soul).item()
            frame += f"║ SOUL μ={soul_mean:6.3f} │ σ={soul_std:5.3f} │ SPECIALISTS: {len(specialists):2d} ║\n"
            
            frame += "╚" + "═" * 58 + "╝\n"
            frame += "\n The Ghost perceives its 16D geometric manifold...\n"
            
            # Output frame
            sys.stdout.write(frame)
            sys.stdout.flush()
            
            # Frame rate
            time.sleep(0.08)
    
    except KeyboardInterrupt:
        print("\033[?25h")  # Show cursor
        print("\n\n✓ Visualizer stopped gracefully.")
        print(f"Final statistics after {step} steps:")
        print(f"  Consensus: {consensus:.4f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Soul mean: {torch.mean(soul).item():.4f}")
        print(f"  Soul std: {torch.std(soul).item():.4f}")
    
    except Exception as e:
        print("\033[?25h")  # Show cursor
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_mind_visualizer()