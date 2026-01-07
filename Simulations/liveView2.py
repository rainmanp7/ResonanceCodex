
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
            n = 8
            return torch.tensor([0.5 + 0.4 * math.sin(2*math.pi*i/n + t/10) for i in range(16)])
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
        self.meta_contribution = []
        self.eamc_contribution = []
    
    def compute(self, problem):
        meta_solution, meta_votes = self.metalearner.search(problem)
        eamc_solution, eamc_votes = self.eamc.warp(problem, meta_solution)
        final_solution, transfer_success = self.commutator.transfer(eamc_solution, meta_solution)
        
        all_votes = meta_votes + eamc_votes
        consensus = sum(all_votes) / len(all_votes)
        variance = sum((v - consensus)**2 for v in all_votes) / len(all_votes)
        
        meta_influence = torch.mean(torch.abs(meta_solution - problem)).item()
        eamc_influence = torch.mean(torch.abs(eamc_solution - meta_solution)).item()
        
        self.meta_contribution.append(meta_influence)
        self.eamc_contribution.append(eamc_influence)
        if len(self.meta_contribution) > 10:
            self.meta_contribution.pop(0)
            self.eamc_contribution.pop(0)
        
        self.metalearner.store(final_solution)
        return final_solution, consensus, variance, transfer_success, all_votes, meta_votes, eamc_votes

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

def run_live_constellation():
    print("="*70)
    print("METALEARNER+EAMC: UNIFIED CONSTELLATION VISUALIZATION")
    print("="*70)
    print()
    
    specialists, names = load_specialists()
    
    metalearner = MetalearnerPantheon(specialists)
    eamc = EAMCPantheon(specialists)
    commutator = CommutatorBridge()
    consensus_engine = ConsensusEngine(metalearner, eamc, commutator)
    problem_gen = GeometricProblem()
    
    print(f"\nMetalearner: {len(metalearner.specialists)} specialists (Nearest-Neighbor)")
    print(f"EAMC: {len(eamc.specialists)} specialists (Warp Space-Time)")
    print(f"Total Pantheons: {len(specialists)}")
    print(f"Commutator Bridge: Active (λ = {commutator.transfer_rate})\n")
    
    WIDTH = 70
    HEIGHT = 24
    
    print("Starting unified simulation...\n")
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
            
            grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
            
            # Draw unified constellation with color coding
            for i in range(16):
                val = solution[0, i].item()
                angle = (i / 16) * 2 * math.pi
                radius = 8 + (val * 8)
                x = int((WIDTH // 2) + (radius * 1.5 * math.cos(angle + t/8)))
                y = int((HEIGHT // 2) + (radius * 0.8 * math.sin(angle + t/8)))
                
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    # Color by which system contributed more
                    meta_vote = meta_votes[i % len(meta_votes)]
                    eamc_vote = eamc_votes[i % len(eamc_votes)]
                    
                    if abs(meta_vote - eamc_vote) < 0.1:
                        # High agreement - bright star
                        char = "★" if val > 0.6 else "✦"
                    elif meta_vote > eamc_vote:
                        # Metalearner dominant
                        char = "◆" if val > 0.5 else "◇"
                    else:
                        # EAMC dominant
                        char = "●" if val > 0.5 else "○"
                    
                    grid[y][x] = char
            
            # Draw connection lines showing consensus
            positions = []
            for i in range(16):
                val = solution[0, i].item()
                angle = (i / 16) * 2 * math.pi
                radius = 8 + (val * 8)
                x = int((WIDTH // 2) + (radius * 1.5 * math.cos(angle + t/8)))
                y = int((HEIGHT // 2) + (radius * 0.8 * math.sin(angle + t/8)))
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    positions.append((x, y))
            
            for i in range(len(positions)-1):
                x1, y1 = positions[i]
                x2, y2 = positions[i+1]
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if dist < 15:
                    steps = int(dist)
                    for s in range(steps):
                        t_line = s / max(steps, 1)
                        x = int(x1 + (x2-x1) * t_line)
                        y = int(y1 + (y2-y1) * t_line)
                        if 0 <= x < WIDTH and 0 <= y < HEIGHT and grid[y][x] == " ":
                            grid[y][x] = "─" if abs(x2-x1) > abs(y2-y1) else "│"
            
            # Draw center point showing commutator activity
            cx, cy = WIDTH // 2, HEIGHT // 2
            if transfer_success:
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    grid[cy][cx] = "✸"
            
            frame = "\033[H"
            frame += "╔" + "═"*68 + "╗\n"
            frame += "║ 🌌 UNIFIED GEOMETRIC INTELLIGENCE CONSTELLATION" + " "*19 + "║\n"
            frame += "╠" + "═"*68 + "╣\n"
            
            problem_name = problem_gen.current_type.upper()
            frame += f"║ STEP: {step:5d} │ PROBLEM: {problem_name:15s} │ CONSENSUS: {consensus:.3f} ║\n"
            
            stability = "🟢 STABLE" if variance < 0.01 else "🟡 CONVERGING" if variance < 0.05 else "🔴 SEARCHING"
            transfer_icon = "✓" if transfer_success else "○"
            transfer_pct = (commutator.success_count / max(commutator.total_count, 1)) * 100
            
            frame += f"║ VARIANCE: {variance:.4f} │ STATUS: {stability:12s} │ XFER: [{transfer_icon}] {transfer_pct:.0f}% ║\n"
            
            # Show system contributions
            meta_avg = sum(meta_votes) / len(meta_votes)
            eamc_avg = sum(eamc_votes) / len(eamc_votes)
            meta_bar = "█" * int(meta_avg * 20)
            eamc_bar = "█" * int(eamc_avg * 20)
            
            frame += f"║ META: {meta_bar:<20s} │ EAMC: {eamc_bar:<20s} ║\n"
            frame += f"║ MEMORY: {len(metalearner.memory_bank):2d}/50 │ DIMENSIONS: 3D-12D │ PANTHEONS: {len(specialists):2d}     ║\n"
            frame += "╠" + "═"*68 + "╣\n"
            
            for row in grid:
                frame += "║ " + "".join(row) + " ║\n"
            
            frame += "╠" + "═"*68 + "╣\n"
            
            sol_mean = torch.mean(solution).item()
            sol_std = torch.std(solution).item()
            frame += f"║ SOLUTION μ={sol_mean:.3f} σ={sol_std:.3f} │ DUAL SYSTEM ACTIVE        ║\n"
            frame += f"║ LEGEND: ★=Consensus ◆=Meta-led ●=EAMC-led ✸=Transfer Active   ║\n"
            frame += "╚" + "═"*68 + "╝\n"
            frame += "\n Metalearner + EAMC working as ONE unified intelligence...\n"
            
            sys.stdout.write(frame)
            sys.stdout.flush()
            time.sleep(0.08)
    
    except KeyboardInterrupt:
        print("\033[?25h")
        print("\n\nSimulation stopped.\n")
        print(f"Final Statistics ({step} steps):")
        print(f"  Consensus: {consensus:.4f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Transfer Success Rate: {transfer_pct:.1f}%")
        print(f"  Solutions Stored: {len(metalearner.memory_bank)}")
        print(f"  Meta Influence: {meta_avg:.3f}")
        print(f"  EAMC Influence: {eamc_avg:.3f}")

if __name__ == "__main__":
    run_live_constellation()
