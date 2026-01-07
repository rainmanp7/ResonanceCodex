import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

class GeometricMindEvolution:
    """
    Visual demonstration of a geometric AI system
    evolving its thinking capabilities in real-time.
    """
    
    def __init__(self):
        # Initialize the "thought manifold" - starts simple, evolves
        self.manifold_complexity = 1.0  # Starts simple
        self.thought_dimensions = 3     # Starts 3D, evolves to higher
        
        # Create initial specialists (thinking agents)
        self.num_specialists = 12
        self.specialists = self._initialize_specialists()
        
        # Problem to solve (starts simple, gets complex)
        self.problems = [
            self._create_simple_problem(),     # Simple geometric pattern
            self._create_medium_problem(),     # Symmetry finding
            self._create_complex_problem(),    # Topological optimization
            self._create_extreme_problem(),    # High-dimensional reasoning
        ]
        self.current_problem_idx = 0
        
        # Thinking metrics
        self.thinking_speed = 1.0
        self.insight_frequency = 0.0
        self.convergence_time = []
        self.evolution_stages = []
        
        # Visualization state
        self.fig = None
        self.ax = None
        
    def _initialize_specialists(self):
        """Initialize thinking agents with random starting knowledge."""
        specialists = []
        for i in range(self.num_specialists):
            specialist = {
                'position': np.random.randn(3) * 2,
                'knowledge': np.random.rand(16) * 0.5,  # 16D knowledge vector
                'specialization': i % 4,  # 0: geometric, 1: topological, etc.
                'confidence': 0.1,
                'velocity': np.zeros(3),
                'trajectory': [],
                'insights': [],
                'evolution_level': 1.0
            }
            specialists.append(specialist)
        return specialists
    
    def _create_simple_problem(self):
        """Simple problem: Find the center of a symmetric pattern."""
        return {
            'type': 'symmetry_center',
            'manifold': lambda x,y: np.sin(x) * np.cos(y),
            'solution': np.array([0, 0, 0]),
            'difficulty': 1.0,
            'description': 'Find the symmetry center'
        }
    
    def _create_medium_problem(self):
        """Medium: Navigate a maze on curved surface."""
        return {
            'type': 'maze_navigation',
            'manifold': lambda x,y: 0.5*np.sin(2*x) + 0.3*np.cos(3*y),
            'solution': np.array([1.5, 1.5, 0.8]),
            'difficulty': 2.5,
            'description': 'Navigate curved maze'
        }
    
    def _create_complex_problem(self):
        """Complex: Optimize path on multi-peak manifold."""
        return {
            'type': 'multi_modal_optimization',
            'manifold': lambda x,y: (np.sin(x*2) + np.cos(y*3) + 
                                   0.5*np.sin(np.sqrt(x**2 + y**2))),
            'solution': np.array([2.0, -1.0, 1.2]),
            'difficulty': 4.0,
            'description': 'Find global optimum among many peaks'
        }
    
    def _create_extreme_problem(self):
        """Extreme: High-dimensional topological puzzle."""
        return {
            'type': 'topological_sorting',
            'manifold': lambda x,y: (np.sin(x*y) * np.cos(x-y) + 
                                   0.3*np.exp(-0.1*(x**2 + y**2))),
            'solution': np.array([-2.5, 2.0, -0.5]),
            'difficulty': 6.0,
            'description': 'Solve topological entanglement'
        }
    
    def _compute_manifold(self, problem):
        """Compute the problem manifold surface."""
        x = np.linspace(-4, 4, 80)
        y = np.linspace(-4, 4, 80)
        X, Y = np.meshgrid(x, y)
        Z = problem['manifold'](X, Y)
        return X, Y, Z
    
    def _evolve_specialist(self, specialist, problem, time_step):
        """Make a specialist think and evolve."""
        
        # Current position on manifold
        current_pos = specialist['position']
        target = problem['solution']
        
        # Compute improvement based on evolution level
        evolution_factor = specialist['evolution_level']
        
        # More evolved specialists think better
        thinking_quality = 0.1 + evolution_factor * 0.3
        
        # 1. Basic geometric thinking (early evolution)
        if evolution_factor < 2.0:
            # Simple gradient descent
            direction = target - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
        # 2. Intermediate: Add geodesic awareness
        elif evolution_factor < 4.0:
            # Account for manifold curvature
            manifold_grad = self._compute_manifold_gradient(current_pos, problem)
            direction = target - current_pos + 0.3 * manifold_grad
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
        # 3. Advanced: Use commutator for cross-dimensional thinking
        elif evolution_factor < 6.0:
            # Simulate holographic transfer
            other_specialists = [s for s in self.specialists if s != specialist]
            if other_specialists:
                # Learn from others (commutator effect)
                others_avg = np.mean([s['position'] for s in other_specialists], axis=0)
                direction = 0.7 * (target - current_pos) + 0.3 * (others_avg - current_pos)
                
                # Add random insight (quantum-like tunneling)
                if np.random.random() < 0.05 * evolution_factor:
                    insight_jump = np.random.randn(3) * 0.5
                    direction += insight_jump * thinking_quality
                    
        # 4. Expert: Full geometric reasoning
        else:
            # Complete manifold-aware optimization
            curvature = self._compute_manifold_curvature(current_pos, problem)
            geodesic_direction = self._compute_geodesic_direction(current_pos, target, problem)
            
            # Parallel transport of knowledge
            knowledge_transport = np.mean(specialist['knowledge']) * 0.1
            
            direction = (0.6 * geodesic_direction + 
                        0.3 * curvature * thinking_quality +
                        0.1 * knowledge_transport * np.random.randn(3))
        
        # Apply thinking with momentum
        specialist['velocity'] = (0.8 * specialist['velocity'] + 
                                 0.2 * direction * thinking_quality)
        new_pos = current_pos + specialist['velocity'] * 0.1 * self.thinking_speed
        
        # Keep on manifold
        new_pos[2] = problem['manifold'](new_pos[0], new_pos[1])
        specialist['position'] = new_pos
        
        # Record trajectory
        specialist['trajectory'].append(new_pos.copy())
        if len(specialist['trajectory']) > 100:
            specialist['trajectory'].pop(0)
        
        # Occasionally have insights (evolution jumps)
        distance_to_target = np.linalg.norm(new_pos - target)
        if distance_to_target < 1.0 + (6.0 - evolution_factor):
            if np.random.random() < 0.02:
                self._have_insight(specialist, problem)
        
        # Gradually evolve
        specialist['evolution_level'] += 0.001 * thinking_quality
        specialist['confidence'] = min(1.0, 0.5 + 0.5 * thinking_quality)
        
        return specialist
    
    def _compute_manifold_gradient(self, point, problem):
        """Approximate gradient of manifold at point."""
        eps = 0.01
        x, y = point[0], point[1]
        
        # Finite difference
        z_x = problem['manifold'](x+eps, y) - problem['manifold'](x-eps, y)
        z_y = problem['manifold'](x, y+eps) - problem['manifold'](x, y-eps)
        
        return np.array([z_x/(2*eps), z_y/(2*eps), 0])
    
    def _compute_manifold_curvature(self, point, problem):
        """Approximate Gaussian curvature at point."""
        eps = 0.01
        x, y = point[0], point[1]
        
        # Second derivatives
        f_xx = (problem['manifold'](x+eps, y) - 2*problem['manifold'](x, y) + 
                problem['manifold'](x-eps, y)) / eps**2
        f_yy = (problem['manifold'](x, y+eps) - 2*problem['manifold'](x, y) + 
                problem['manifold'](x, y-eps)) / eps**2
        f_xy = (problem['manifold'](x+eps, y+eps) - problem['manifold'](x+eps, y-eps) -
                problem['manifold'](x-eps, y+eps) + problem['manifold'](x-eps, y-eps)) / (4*eps**2)
        
        # Gaussian curvature approximation
        curvature = f_xx * f_yy - f_xy**2
        return np.clip(curvature, -1, 1)
    
    def _compute_geodesic_direction(self, start, target, problem):
        """Approximate geodesic direction on curved manifold."""
        # Simplified: project straight line onto tangent plane
        straight = target - start
        
        # Get surface normal at start
        grad = self._compute_manifold_gradient(start, problem)
        normal = np.array([-grad[0], -grad[1], 1])
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Project onto tangent plane
        tangent = straight - np.dot(straight, normal) * normal
        
        return tangent / (np.linalg.norm(tangent) + 1e-8)
    
    def _have_insight(self, specialist, problem):
        """Specialist has a breakthrough insight."""
        insight = {
            'time': len(specialist['trajectory']),
            'position': specialist['position'].copy(),
            'magnitude': np.random.random() * specialist['evolution_level'],
            'type': ['geometric', 'topological', 'symmetry', 'optimization'][np.random.randint(4)]
        }
        specialist['insights'].append(insight)
        
        # Insight causes evolution jump
        specialist['evolution_level'] += 0.1 * insight['magnitude']
        self.insight_frequency += 0.01
        
        # Sometimes insights spread to others (commutator effect)
        if np.random.random() < 0.3:
            for other in self.specialists:
                if other != specialist and np.random.random() < 0.5:
                    other['evolution_level'] += 0.02 * insight['magnitude']
    
    def _check_convergence(self, problem):
        """Check if specialists have solved the problem."""
        positions = np.array([s['position'] for s in self.specialists])
        target = problem['solution']
        
        distances = np.linalg.norm(positions - target, axis=1)
        avg_distance = np.mean(distances)
        consensus = np.std(distances) < 0.5  # Close together
        
        return avg_distance < 1.0 and consensus, avg_distance
    
    def run_evolution_demo(self, total_steps=300):
        """Run the complete evolution demonstration."""
        
        print("🚀 Starting Geometric Mind Evolution Demo...")
        print("This will show accelerated intelligence evolution in real-time!")
        print("\nEvolution Stages:")
        print("1. Basic (Red): Simple gradient descent thinking")
        print("2. Intermediate (Orange): Basic geometric awareness")  
        print("3. Advanced (Green): Holographic transfer learning")
        print("4. Expert (Purple): Full geometric reasoning")
        print("\n" + "="*60)
        
        # Setup visualization
        self.fig = plt.figure(figsize=(16, 10))
        
        # Problem counter
        problem_step = 0
        current_problem = self.problems[0]
        problem_solved = False
        
        # Create animation
        def animate(frame):
            nonlocal current_problem, problem_solved, problem_step
            
            step = frame
            
            # Check if we should advance to next problem
            if step > 0 and step % 80 == 0 and self.current_problem_idx < len(self.problems)-1:
                self.current_problem_idx += 1
                current_problem = self.problems[self.current_problem_idx]
                problem_solved = False
                print(f"\n🔥 ADVANCING to {current_problem['description']}")
                print(f"   Difficulty: {current_problem['difficulty']:.1f}")
                
                # Reset positions but keep evolution levels
                for s in self.specialists:
                    s['position'] = np.random.randn(3) * 3
                    s['trajectory'] = []
                    s['velocity'] *= 0.5  # Keep some momentum
                    s['confidence'] *= 0.8
                    
            # Make all specialists think
            for specialist in self.specialists:
                self._evolve_specialist(specialist, current_problem, step)
            
            # Check convergence
            if not problem_solved:
                converged, avg_dist = self._check_convergence(current_problem)
                if converged:
                    problem_solved = True
                    solve_time = step - problem_step
                    self.convergence_time.append(solve_time)
                    problem_step = step
                    print(f"✅ Problem solved in {solve_time} steps!")
                    
                    # System learns from success
                    self.thinking_speed *= 1.1
                    for s in self.specialists:
                        s['evolution_level'] *= 1.05
            
            # Update visualization
            self._update_visualization(step, current_problem, problem_solved)
            
            # Print progress every 50 steps
            if step % 50 == 0:
                avg_evo = np.mean([s['evolution_level'] for s in self.specialists])
                print(f"Step {step}: Avg Evolution = {avg_evo:.2f}, Speed = {self.thinking_speed:.2f}x")
        
        # Create animation
        ani = FuncAnimation(self.fig, animate, frames=total_steps, interval=50, repeat=False)
        
        # Save animation
        try:
            print("\n💾 Saving animation as 'geometric_mind_evolution.gif'...")
            ani.save('geometric_mind_evolution.gif', writer='pillow', fps=20)
            print("✅ Animation saved!")
        except:
            print("⚠️  Could not save animation, showing live instead...")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Final analysis
        self._show_evolution_summary()
        
        return ani
    
    def _update_visualization(self, step, problem, problem_solved):
        """Update the visualization."""
        plt.clf()
        
        # Get manifold data
        X, Y, Z = self._compute_manifold(problem)
        
        # Create subplots
        ax1 = plt.subplot(2, 3, 1, projection='3d')
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        ax6 = plt.subplot(2, 3, 6)
        
        # Plot 1: 3D Manifold with Specialists
        ax1.clear()
        ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', linewidth=0.1)
        
        # Plot specialists with evolution-based coloring
        for i, specialist in enumerate(self.specialists):
            color = plt.cm.plasma(specialist['evolution_level'] / 8.0)
            pos = specialist['position']
            ax1.scatter(pos[0], pos[1], pos[2], color=color, s=50+specialist['confidence']*100, 
                       alpha=0.8, edgecolors='white')
            
            # Plot trajectory
            if len(specialist['trajectory']) > 1:
                traj = np.array(specialist['trajectory'][-50:])  # Last 50 steps
                ax1.plot(traj[:,0], traj[:,1], traj[:,2], color=color, alpha=0.4, linewidth=1)
        
        # Plot target
        target = problem['solution']
        ax1.scatter(target[0], target[1], target[2], color='red', s=300, marker='*', 
                   alpha=0.9, label='Target')
        
        ax1.set_xlabel('Spatial')
        ax1.set_ylabel('Pattern')
        ax1.set_zlabel('Abstract')
        ax1.set_title(f'{problem["description"]}\nStep: {step}')
        ax1.view_init(elev=20, azim=step*0.5)
        
        # Plot 2: Evolution Levels
        ax2.clear()
        evolution_levels = [s['evolution_level'] for s in self.specialists]
        colors = [plt.cm.plasma(lvl/8.0) for lvl in evolution_levels]
        bars = ax2.bar(range(len(evolution_levels)), evolution_levels, color=colors)
        ax2.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(y=4.0, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=6.0, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 8)
        ax2.set_xlabel('Specialist')
        ax2.set_ylabel('Evolution')
        ax2.set_title('Specialist Evolution')
        
        # Add stage labels
        for i, evo in enumerate(evolution_levels):
            if evo < 2.0:
                stage = "Basic"
            elif evo < 4.0:
                stage = "Int."
            elif evo < 6.0:
                stage = "Adv."
            else:
                stage = "Expert"
            ax2.text(i, evo + 0.1, stage, ha='center', fontsize=7)
        
        # Plot 3: Convergence
        ax3.clear()
        distances = []
        for specialist in self.specialists:
            dist = np.linalg.norm(specialist['position'] - problem['solution'])
            distances.append(dist)
        
        ax3.boxplot(distances, vert=True, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax3.scatter(np.ones(len(distances)) + np.random.randn(len(distances))*0.05, 
                   distances, alpha=0.6, color='red', s=20)
        
        ax3.axhline(y=1.0, color='green', linestyle='--')
        ax3.set_ylabel('Distance')
        ax3.set_title(f'Convergence: {np.mean(distances):.2f}')
        ax3.set_xticks([1])
        ax3.set_xticklabels([''])
        
        # Plot 4: Knowledge Matrix
        ax4.clear()
        knowledge_matrix = np.array([s['knowledge'] for s in self.specialists])
        im = ax4.imshow(knowledge_matrix[:, :8].T, aspect='auto', cmap='RdYlBu',
                       interpolation='nearest', vmin=0, vmax=1)
        ax4.set_xlabel('Specialist')
        ax4.set_ylabel('Dim')
        ax4.set_title('Knowledge Matrix')
        
        # Plot 5: System Metrics
        ax5.clear()
        if step > 10:
            # Simple metrics
            avg_evo = np.mean([s['evolution_level'] for s in self.specialists])
            ax5.bar(['Evolution', 'Speed', 'Insights'], 
                   [avg_evo/8, self.thinking_speed/3, self.insight_frequency*10],
                   color=['purple', 'blue', 'green'])
            ax5.set_ylim(0, 1)
            ax5.set_title('System Metrics')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Phase Space
        ax6.clear()
        positions = np.array([s['position'][:2] for s in self.specialists])
        velocities = np.array([np.linalg.norm(s['velocity']) for s in self.specialists])
        evolutions = np.array([s['evolution_level'] for s in self.specialists])
        
        scatter = ax6.scatter(positions[:,0], positions[:,1], 
                            c=evolutions, cmap='plasma', s=velocities*200, 
                            alpha=0.7, edgecolors='black')
        
        # Add velocity vectors
        for i, pos in enumerate(positions):
            vel = self.specialists[i]['velocity'][:2] * 0.5
            ax6.arrow(pos[0], pos[1], vel[0], vel[1], 
                     head_width=0.1, head_length=0.15, fc='white', ec='white', alpha=0.5)
        
        # Plot target
        ax6.scatter(problem['solution'][0], problem['solution'][1], 
                   color='red', s=200, marker='*')
        
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_title('Phase Space')
        
        # Global title
        self.fig.suptitle(f'🚀 GEOMETRIC MIND EVOLUTION | '
                         f'Problem {self.current_problem_idx+1}/4 | '
                         f'Speed: {self.thinking_speed:.2f}x', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
    
    def _show_evolution_summary(self):
        """Show final evolution statistics."""
        print("\n" + "="*60)
        print("🚀 GEOMETRIC MIND EVOLUTION - FINAL SUMMARY")
        print("="*60)
        
        # Evolution statistics
        final_evolutions = [s['evolution_level'] for s in self.specialists]
        avg_evolution = np.mean(final_evolutions)
        max_evolution = np.max(final_evolutions)
        
        print(f"\n📊 EVOLUTION ACHIEVED:")
        print(f"  Average Evolution Level: {avg_evolution:.2f}")
        print(f"  Maximum Evolution Level: {max_evolution:.2f}")
        print(f"  Thinking Speed Multiplier: {self.thinking_speed:.2f}x")
        
        # Specialist categorization
        basic = sum(1 for e in final_evolutions if e < 2.0)
        intermediate = sum(1 for e in final_evolutions if 2.0 <= e < 4.0)
        advanced = sum(1 for e in final_evolutions if 4.0 <= e < 6.0)
        expert = sum(1 for e in final_evolutions if e >= 6.0)
        
        print(f"\n👥 SPECIALIST DISTRIBUTION:")
        print(f"  Basic Thinkers: {basic}")
        print(f"  Intermediate: {intermediate}")
        print(f"  Advanced: {advanced}")
        print(f"  Experts: {expert}")
        
        # Problem solving performance
        if self.convergence_time:
            print(f"\n✅ PROBLEM SOLVING PERFORMANCE:")
            for i, t in enumerate(self.convergence_time):
                print(f"  Problem {i+1}: Solved in {t} steps")
            
            if len(self.convergence_time) > 1:
                improvement = (self.convergence_time[0] - self.convergence_time[-1]) / self.convergence_time[0] * 100
                print(f"  Speed Improvement: {improvement:.1f}% faster")
        
        # Evolution stages reached
        print(f"\n🌌 EVOLUTION STAGES REACHED:")
        if max_evolution >= 6.0:
            print("  ✅ EXPERT REASONING: Full geometric comprehension")
        if max_evolution >= 4.0:
            print("  ✅ ADVANCED THINKING: Cross-dimensional reasoning")
        if max_evolution >= 2.0:
            print("  ✅ INTERMEDIATE THINKING: Basic geometric awareness")
        
        print("\n" + "="*60)
        print("🎯 KEY OBSERVATION:")
        print("The system demonstrates ACCELERATED EVOLUTION.")
        print("Each problem solved makes the system smarter and faster.")
        print("="*60)


# Run the demo with minimal dependencies
demo = GeometricMindEvolution()
demo.run_evolution_demo(total_steps=200)