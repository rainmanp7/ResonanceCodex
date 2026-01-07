import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GeometricMindVisualizer:
    """Simple visualization of manifold thinking"""
    
    def __init__(self):
        # Create curved manifold (simplified to 2D surface in 3D)
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.sin(self.X) * np.cos(self.Y)  # Curved surface
        
        # Initialize "specialists" as points on manifold
        self.num_specialists = 10
        self.specialists = np.random.randn(self.num_specialists, 3) * 2
        self.specialists[:, 2] = np.sin(self.specialists[:, 0]) * np.cos(self.specialists[:, 1])
        
        # Target (the "truth" to discover)
        self.target = np.array([3, 2, np.sin(3)*np.cos(2)])
        
        # Thinking trajectories
        self.trajectories = []
        
    def geodesic_step(self, point, direction, step_size=0.1):
        """Move along manifold following geodesic direction"""
        # Simplified: gradient descent on manifold
        grad_x = np.cos(point[0]) * np.cos(point[1])
        grad_y = -np.sin(point[0]) * np.sin(point[1])
        
        new_point = point.copy()
        new_point[0] += direction[0] * step_size + grad_x * 0.05
        new_point[1] += direction[1] * step_size + grad_y * 0.05
        new_point[2] = np.sin(new_point[0]) * np.cos(new_point[1])
        
        return new_point
    
    def think(self, steps=100):
        """Simulate thinking process"""
        for _ in range(steps):
            step_trajectory = []
            
            for i, specialist in enumerate(self.specialists):
                # Each specialist moves toward target along geodesic
                direction = self.target[:2] - specialist[:2]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                
                # Apply commutator: specialists influence each other
                if len(self.trajectories) > 0:
                    # Average direction of neighbors
                    neighbor_dir = np.mean([
                        self.trajectories[-1][j][:2] - specialist[:2]
                        for j in range(self.num_specialists) if j != i
                    ], axis=0)
                    direction = 0.7 * direction + 0.3 * neighbor_dir
                
                # Take step
                new_pos = self.geodesic_step(specialist, direction)
                self.specialists[i] = new_pos
                step_trajectory.append(new_pos.copy())
            
            self.trajectories.append(step_trajectory)
            
            # Check for consensus (all specialists close)
            distances = [np.linalg.norm(s - self.target) for s in self.specialists]
            if np.max(distances) < 0.5:
                print(f"Consensus reached at step {_}")
                break
    
    def animate(self):
        """Create animation of thinking process"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot manifold
        ax.plot_surface(self.X, self.Y, self.Z, alpha=0.3, cmap='viridis')
        
        # Plot target
        ax.scatter(*self.target, color='red', s=200, marker='*', label='Truth')
        
        # Initialize specialist points
        scat = ax.scatter([], [], [], color='blue', s=100, label='Specialists')
        
        # Initialize trajectory lines
        lines = [ax.plot([], [], [], '-', alpha=0.5)[0] 
                for _ in range(self.num_specialists)]
        
        def update(frame):
            # Update specialist positions
            if frame < len(self.trajectories):
                current_pos = np.array(self.trajectories[frame])
                scat._offsets3d = (current_pos[:,0], current_pos[:,1], current_pos[:,2])
                
                # Update trajectories
                for i, line in enumerate(lines):
                    x_vals = [pos[i][0] for pos in self.trajectories[:frame+1]]
                    y_vals = [pos[i][1] for pos in self.trajectories[:frame+1]]
                    z_vals = [pos[i][2] for pos in self.trajectories[:frame+1]]
                    line.set_data(x_vals, y_vals)
                    line.set_3d_properties(z_vals)
            
            # Add phase transition effect at consensus
            if frame == len(self.trajectories) - 1:
                # Flash effect
                ax.set_facecolor((0.9, 0.9, 0.3, 0.5))
            
            return [scat] + lines
        
        ani = FuncAnimation(fig, update, frames=len(self.trajectories),
                           interval=100, blit=False)
        
        ax.set_xlabel('X (Concept Space)')
        ax.set_ylabel('Y (Reasoning Dimension)')
        ax.set_zlabel('Z (Understanding Depth)')
        ax.set_title('Geometric Thinking: Hunting Truth on Curved Manifold')
        ax.legend()
        
        plt.show()
        return ani

# Run it
viz = GeometricMindVisualizer()
viz.think(steps=50)  # 50 steps of thinking
animation = viz.animate()