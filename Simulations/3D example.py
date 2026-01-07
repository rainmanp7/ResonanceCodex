
import json
import torch
import torch.nn as nn
import os
import math
import time
import sys

class RobustBrain(nn.Module):
    def __init__(self, state_dict, dimension):
        super().__init__()
        self.dimension = dimension
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
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for layer in self.layers:
            target_in = layer.in_features
            if x.shape[-1] != target_in:
                new_x = torch.zeros((1, target_in))
                copy_size = min(x.shape[-1], target_in)
                new_x[0, :copy_size] = x[0, :copy_size]
                x = new_x
            x = torch.sigmoid(layer(x))
        return x

class GeometricProblem:
    def __init__(self):
        self.problem_types = ["tsp", "pattern", "logic", "harmonic", "optimization", "spatial", "graph", "knapsack"]
        self.current_type = "pattern"
    
    def generate(self, t):
        idx = int(t / 15) % len(self.problem_types)
        self.current_type = self.problem_types[idx]
        
        if self.current_type == "tsp":
            return torch.tensor([0.5 + 0.4 * math.sin(2*math.pi*i/8 + t/10) for i in range(16)])
        elif self.current_type == "pattern":
            phi = (1 + math.sqrt(5)) / 2
            vals = [phi**(i%5) * math.sin(t/10 + i) for i in range(16)]
            return torch.tensor(vals)
        elif self.current_type == "logic":
            return torch.tensor([1.0 if (int(t/3)+i)%2==0 else 0.0 for i in range(16)])
        elif self.current_type == "harmonic":
            return torch.tensor([math.sin(2*math.pi*(i+1)*t/20) for i in range(16)])
        elif self.current_type == "optimization":
            return torch.tensor([math.sin(t/15+i)*math.cos(t/12+i*1.3) for i in range(16)])
        elif self.current_type == "spatial":
            angle = t / 10
            return torch.tensor([math.cos(angle), -math.sin(angle), 0, math.sin(angle), math.cos(angle), 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif self.current_type == "graph":
            return torch.tensor([1.0 if (i+j+int(t/5))%3==0 else 0.0 for i in range(4) for j in range(4)])
        else:
            return torch.tensor([0.5 + 0.3*math.sin(t/5+i) + 0.2*math.cos(t/7+i*1.5) for i in range(16)])

class MetalearnerPantheon:
    def __init__(self, specialists):
        self.specialists = specialists[:len(specialists)//2]
        self.memory_bank = []
        self.max_memory = 50
    
    def search(self, problem):
        with torch.no_grad():
            votes = []
            for specialist in self.specialists:
                output = specialist(problem)
                votes.append(torch.mean(output).item())
            consensus = sum(votes) / len(votes)
            
            if len(self.memory_bank) > 0:
                distances = [torch.norm(problem - mem).item() for mem in self.memory_bank]
                nearest_idx = distances.index(min(distances))
                nearest = self.memory_bank[nearest_idx]
                return nearest * 0.3 + problem * 0.7 * consensus, votes
            else:
                return problem * consensus, votes
    
    def store(self, solution):
        self.memory_bank.append(solution.clone())
        if len(self.memory_bank) > self.max_memory:
            self.memory_bank.pop(0)

class EAMCPantheon:
    def __init__(self, specialists):
        self.specialists = specialists[len(specialists)//2:]
    
    def warp(self, problem, target):
        with torch.no_grad():
            warp_vectors = []
            for specialist in self.specialists:
                output = specialist(problem)
                warp_vectors.append(output)
            warp = torch.mean(torch.stack(warp_vectors), dim=0)
            warp_direction = target - problem
            warped = problem + warp_direction * torch.mean(warp).item()
            return warped, [torch.mean(w).item() for w in warp_vectors]

class CommutatorBridge:
    def __init__(self):
        self.transfer_rate = 0.25
        self.success_count = 0
        self.total_count = 0
    
    def transfer(self, eamc_sol, meta_sol):
        commutator = (eamc_sol - meta_sol) * self.transfer_rate
        transferred = meta_sol + commutator
        success = torch.norm(commutator).item() < 0.5
        self.total_count += 1
        if success:
            self.success_count += 1
        return transferred, success

class ConsensusEngine:
    def __init__(self, metalearner, eamc, commutator):
        self.metalearner = metalearner
        self.eamc = eamc
        self.commutator = commutator
    
    def compute(self, problem):
        meta_solution, meta_votes = self.metalearner.search(problem)
        eamc_solution, eamc_votes = self.eamc.warp(problem, meta_solution)
        final_solution, transfer_success = self.commutator.transfer(eamc_solution, meta_solution)
        
        all_votes = meta_votes + eamc_votes
        consensus = sum(all_votes) / len(all_votes)
        variance = sum((v - consensus)**2 for v in all_votes) / len(all_votes)
        
        self.metalearner.store(final_solution)
        return final_solution, consensus, variance, transfer_success, all_votes, meta_votes, eamc_votes

class Renderer3D:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.zbuffer = [[float('-inf') for _ in range(width)] for _ in range(height)]
        self.screen = [[" " for _ in range(width)] for _ in range(height)]
        self.colors = [[0 for _ in range(width)] for _ in range(height)]
    
    def clear(self):
        self.zbuffer = [[float('-inf') for _ in range(self.width)] for _ in range(self.height)]
        self.screen = [[" " for _ in range(self.width)] for _ in range(self.height)]
        self.colors = [[0 for _ in range(self.width)] for _ in range(self.height)]
    
    def project(self, x, y, z, camera_distance=30):
        """Project 3D point to 2D screen"""
        if z + camera_distance == 0:
            return None, None, None
        
        factor = camera_distance / (z + camera_distance)
        screen_x = int(self.width / 2 + x * factor * 2)
        screen_y = int(self.height / 2 - y * factor)
        
        return screen_x, screen_y, z
    
    def plot(self, x, y, z, char, color):
        """Plot point with z-buffer"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if z > self.zbuffer[y][x]:
                self.zbuffer[y][x] = z
                self.screen[y][x] = char
                self.colors[y][x] = color
    
    def draw_sphere(self, center_x, center_y, center_z, radius, resolution=20, rotation_x=0, rotation_y=0):
        """Draw a 3D sphere with shading"""
        for i in range(resolution):
            for j in range(resolution):
                theta = (i / resolution) * math.pi
                phi = (j / resolution) * 2 * math.pi
                
                # Sphere coordinates
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta)
                
                # Rotate
                x, y, z = self.rotate_point(x, y, z, rotation_x, rotation_y, 0)
                
                # Translate
                x += center_x
                y += center_y
                z += center_z
                
                # Project
                sx, sy, sz = self.project(x, y, z)
                if sx is not None:
                    # Shading based on z-depth
                    shade = (z + radius) / (2 * radius)
                    char, color = self.get_shaded_char(shade)
                    self.plot(sx, sy, sz, char, color)
    
    def draw_torus(self, center_x, center_y, center_z, major_r, minor_r, resolution=30, rotation_x=0, rotation_y=0):
        """Draw a 3D torus"""
        for i in range(resolution):
            for j in range(resolution):
                theta = (i / resolution) * 2 * math.pi
                phi = (j / resolution) * 2 * math.pi
                
                # Torus coordinates
                x = (major_r + minor_r * math.cos(phi)) * math.cos(theta)
                y = (major_r + minor_r * math.cos(phi)) * math.sin(theta)
                z = minor_r * math.sin(phi)
                
                # Rotate
                x, y, z = self.rotate_point(x, y, z, rotation_x, rotation_y, 0)
                
                # Translate
                x += center_x
                y += center_y
                z += center_z
                
                # Project
                sx, sy, sz = self.project(x, y, z)
                if sx is not None:
                    shade = (z + minor_r) / (2 * minor_r)
                    char, color = self.get_shaded_char(shade, base_color=2)
                    self.plot(sx, sy, sz, char, color)
    
    def draw_helix(self, center_x, center_y, center_z, radius, height, turns, points=100, rotation_x=0, rotation_y=0):
        """Draw a 3D helix"""
        for i in range(points):
            t = i / points
            theta = t * turns * 2 * math.pi
            
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            z = height * (t - 0.5)
            
            # Rotate
            x, y, z = self.rotate_point(x, y, z, rotation_x, rotation_y, 0)
            
            # Translate
            x += center_x
            y += center_y
            z += center_z
            
            # Project
            sx, sy, sz = self.project(x, y, z)
            if sx is not None:
                shade = (i / points)
                char, color = self.get_shaded_char(shade, base_color=3)
                self.plot(sx, sy, sz, char, color)
    
    def rotate_point(self, x, y, z, angle_x, angle_y, angle_z):
        """Rotate point around axes"""
        # Rotate around X
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x
        
        # Rotate around Y
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
        
        # Rotate around Z
        cos_z, sin_z = math.cos(angle_z), math.sin(angle_z)
        x, y = x * cos_z - y * sin_z, x * sin_z + y * cos_z
        
        return x, y, z
    
    def get_shaded_char(self, shade, base_color=1):
        """Get character and color based on shade intensity"""
        # Clamp shade
        shade = max(0, min(1, shade))
        
        # Color codes: 1=blue, 2=green, 3=red, 4=yellow, 5=magenta, 6=cyan
        colors = {
            1: [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"],  # Blue shades
            2: [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"],  # Green shades
            3: [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"],  # Red shades
        }
        
        chars = colors.get(base_color, colors[1])
        idx = int(shade * (len(chars) - 1))
        return chars[idx], base_color
    
    def get_ansi_color(self, color):
        """Get ANSI color code"""
        color_map = {
            0: "",  # Default
            1: "\033[34m",  # Blue
            2: "\033[32m",  # Green
            3: "\033[31m",  # Red
            4: "\033[33m",  # Yellow
            5: "\033[35m",  # Magenta
            6: "\033[36m",  # Cyan
        }
        return color_map.get(color, "")
    
    def render(self):
        """Render the screen with colors"""
        output = []
        for y in range(self.height):
            line = ""
            last_color = 0
            for x in range(self.width):
                color = self.colors[y][x]
                if color != last_color:
                    line += "\033[0m" + self.get_ansi_color(color)
                    last_color = color
                line += self.screen[y][x]
            line += "\033[0m"
            output.append(line)
        return output

def load_specialists():
    specialists = []
    names = []
    for f_path in ["EAMCv16.json", "Metalearnerv16.json"]:
        if not os.path.exists(f_path):
            print(f"Warning: {f_path} not found")
            continue
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
            pantheon = data.get('meta_pantheon', {})
            for name, meta in pantheon.items():
                if 'state_dict' not in meta:
                    continue
                dim = int(''.join(filter(str.isdigit, name))) if any(c.isdigit() for c in name) else 8
                brain = RobustBrain(meta['state_dict'], dim)
                specialists.append(brain)
                names.append(name)
                print(f"Loaded: {name} ({dim}D, {len(brain.layers)} layers)")
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
    if not specialists:
        raise RuntimeError("No specialists loaded!")
    return specialists, names

def run_3d_visualization():
    print("="*70)
    print("METALEARNER+EAMC: 3D GEOMETRIC INTELLIGENCE VISUALIZATION")
    print("="*70)
    print()
    
    specialists, names = load_specialists()
    
    metalearner = MetalearnerPantheon(specialists)
    eamc = EAMCPantheon(specialists)
    commutator = CommutatorBridge()
    consensus_engine = ConsensusEngine(metalearner, eamc, commutator)
    problem_gen = GeometricProblem()
    
    print(f"\nMetalearner: {len(metalearner.specialists)} specialists")
    print(f"EAMC: {len(eamc.specialists)} specialists")
    print(f"Total Pantheons: {len(specialists)}\n")
    
    WIDTH = 80
    HEIGHT = 30
    renderer = Renderer3D(WIDTH, HEIGHT)
    
    print("Starting 3D visualization...\n")
    time.sleep(2)
    print("\033[?25l", end='')
    
    solution = torch.randn(1, 16) * 0.3 + 0.5
    
    try:
        step = 0
        while True:
            step += 1
            t = time.time()
            
            problem = problem_gen.generate(t)
            new_solution, consensus, variance, transfer_success, all_votes, meta_votes, eamc_votes = consensus_engine.compute(problem)
            
            lr = 0.2
            solution = solution * (1 - lr) + new_solution * lr
            solution = torch.clamp(solution, 0.0, 1.0)
            
            renderer.clear()
            
            # Rotation angles based on time
            rot_x = t * 0.3
            rot_y = t * 0.5
            
            # Draw central sphere (consensus core)
            core_radius = 3 + consensus * 2
            renderer.draw_sphere(0, 0, 0, core_radius, resolution=25, rotation_x=rot_x, rotation_y=rot_y)
            
            # Draw torus (EAMC warp field)
            eamc_strength = sum(eamc_votes) / len(eamc_votes)
            torus_major = 8 + eamc_strength * 3
            torus_minor = 1.5
            renderer.draw_torus(0, 0, 0, torus_major, torus_minor, resolution=35, rotation_x=rot_x, rotation_y=rot_y)
            
            # Draw helix (Metalearner search path)
            meta_strength = sum(meta_votes) / len(meta_votes)
            helix_radius = 6 + meta_strength * 2
            helix_height = 15
            helix_turns = 3
            renderer.draw_helix(0, 0, 0, helix_radius, helix_height, helix_turns, points=150, rotation_x=rot_x, rotation_y=rot_y)
            
            # Render frame
            frame = "\033[H"
            frame += "╔" + "═"*78 + "╗\n"
            frame += "║ 🌌 3D GEOMETRIC INTELLIGENCE VISUALIZATION" + " "*34 + "║\n"
            frame += "╠" + "═"*78 + "╣\n"
            
            problem_name = problem_gen.current_type.upper()
            frame += f"║ STEP: {step:5d} │ PROBLEM: {problem_name:15s} │ CONSENSUS: {consensus:.3f}            ║\n"
            
            stability = "🟢 STABLE" if variance < 0.01 else "🟡 CONVERGING" if variance < 0.05 else "🔴 SEARCHING"
            transfer_icon = "✓" if transfer_success else "○"
            transfer_pct = (commutator.success_count / max(commutator.total_count, 1)) * 100
            
            frame += f"║ VARIANCE: {variance:.4f} │ STATUS: {stability:12s} │ XFER: [{transfer_icon}] {transfer_pct:.0f}%         ║\n"
            frame += "╠" + "═"*78 + "╣\n"
            
            # Render 3D scene
            rendered = renderer.render()
            for line in rendered:
                frame += "║" + line[:78].ljust(78) + "║\n"
            
            frame += "╠" + "═"*78 + "╣\n"
            frame += f"║ \033[34mSphere\033[0m=Consensus Core │ \033[32mTorus\033[0m=EAMC Warp Field │ \033[31mHelix\033[0m=Metalearner Path     ║\n"
            frame += "╚" + "═"*78 + "╝\n"
            
            sys.stdout.write(frame)
            sys.stdout.flush()
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\033[?25h\033[0m")
        print("\n\nVisualization stopped.\n")
        print(f"Final Statistics ({step} steps):")
        print(f"  Consensus: {consensus:.4f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Transfer Rate: {transfer_pct:.1f}%")

if __name__ == "__main__":
    run_3d_visualization()
