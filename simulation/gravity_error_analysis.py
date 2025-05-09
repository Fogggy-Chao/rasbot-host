import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
from pathlib import Path
from kinematics import RobotKinematics

class GravityAwareErrorAnalysis(RobotKinematics):
    """
    Error analysis that specifically accounts for gravity effects on the robot arm.
    """
    def __init__(self, link_lengths=None):
        """Initialize with robot parameters"""
        super().__init__(link_lengths)
        
        # Gravity effect parameters
        self.gravity_params = {
            'joint_deflection': [0, 0.5, 2.5, 1.0, 0.5],  # Degrees of deflection at each joint
            'load_factor': 0.4,                           # How much load affects deflection
            'distance_factor': 0.15                       # How joint deflection increases with distance from base
        }
    
    def apply_gravity_effects(self, joint_angles):
        """
        Apply gravity effects to joint angles based on arm configuration
        
        Args:
            joint_angles: Original joint angles [degrees]
        
        Returns:
            gravity_affected_angles: Joint angles with gravity effects applied
        """
        # Create a copy of the joint angles
        affected_angles = joint_angles.copy()
        
        # Calculate load factors based on arm extension
        # The more extended the arm, the more gravity effects increase
        extension = 0
        for i in range(1, 4):  # Shoulder, elbow, wrist - most affected by gravity
            # How horizontal is this joint? (0° and 180° = horizontal, 90° = vertical)
            horizontal_factor = abs(np.sin(np.radians(joint_angles[i] % 180)))
            
            # Calculate gravity effect - strongest when arm is extended horizontally
            gravity_effect = self.gravity_params['joint_deflection'][i] * horizontal_factor
            
            # Add load effect (more effect when arm is extended)
            gravity_effect += extension * self.gravity_params['load_factor']
            
            # Apply gravity (negative because gravity pulls downward)
            affected_angles[i] -= gravity_effect
            
            # Accumulate extension effect for next joints
            extension += horizontal_factor * self.gravity_params['distance_factor']
        
        return affected_angles
    
    def generate_test_configurations(self):
        """
        Generate realistic test configurations focused on 4 main joint angles
        
        Returns:
            List of joint configurations
        """
        # Focus on realistic configurations similar to the experiment
        configs = [
            # The configuration from the image (approximately)
            [0, 0, 0, 0, 0, 67],
            
            # Additional realistic configurations
            [0, 30, 0, 0, 0, 67],
            [0, 60, 0, 0, 0, 67],
            [0, 90, 0, 0, 0, 67],
            [0, 0, 30, 0, 0, 67],
            [0, 0, 60, 0, 0, 67],
            [0, 0, 90, 0, 0, 67],
            [0, 0, 0, 30, 0, 67],
            [0, 0, 0, 60, 0, 67],
            [0, 30, 30, 0, 0, 67],
            [0, 60, 60, 0, 0, 67],
            [0, 30, 60, 30, 0, 67],
            [45, 30, 0, 0, 0, 67],
            [90, 30, 0, 0, 0, 67],
            [45, 45, 45, 0, 0, 67],
        ]
        
        return configs
    
    def run_error_analysis(self):
        """
        Run comprehensive error analysis with gravity effects
        
        Returns:
            DataFrame with all results
        """
        # Generate test configurations
        configs = self.generate_test_configurations()
        
        # Prepare results storage
        results = []
        
        # Process each configuration
        for i, config in enumerate(configs):
            print(f"Analyzing configuration {i+1}/{len(configs)}: {[round(a, 1) for a in config]}")
            
            # 1. Calculate ideal position (perfect simulation)
            fk_ideal = self.forward_kinematics(config)
            ideal_position = fk_ideal['positions'][-1]
            
            # 2. Calculate position with gravity effects
            gravity_affected_angles = self.apply_gravity_effects(config)
            fk_gravity = self.forward_kinematics(gravity_affected_angles)
            gravity_position = fk_gravity['positions'][-1]
            
            # 3. Calculate position error
            error_vector = np.array(gravity_position) - np.array(ideal_position)
            euclidean_error = np.linalg.norm(error_vector)
            
            # 4. Store results
            results.append({
                'config_id': i,
                'joint_angles': config,
                'gravity_affected_angles': gravity_affected_angles,
                'ideal_position': ideal_position,
                'gravity_position': gravity_position,
                'euclidean_error': euclidean_error,
                'x_error': error_vector[0],
                'y_error': error_vector[1],
                'z_error': error_vector[2],
                'angle_deflection': [round(config[j] - gravity_affected_angles[j], 2) for j in range(len(config))]
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        return results_df
    
    def visualize_gravity_effects(self, results_df, output_dir=None):
        """
        Create visualizations of gravity effects
        
        Args:
            results_df: DataFrame with analysis results
            output_dir: Directory to save visualizations
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up plot style
        plt.style.use('ggplot')
        
        # 1. Position error due to gravity
        plt.figure(figsize=(10, 6))
        
        # Sort by error magnitude for better visualization
        sorted_df = results_df.sort_values('euclidean_error')
        
        # Create bar chart of position errors
        plt.bar(range(len(sorted_df)), sorted_df['euclidean_error'], color='#3498db', alpha=0.8)
        plt.axhline(sorted_df['euclidean_error'].mean(), color='r', linestyle='--', 
                    label=f'Mean error: {sorted_df["euclidean_error"].mean():.2f} cm')
        
        # Add labels and title
        plt.title('End-Effector Position Error Due to Gravity', fontsize=14, fontweight='bold')
        plt.xlabel('Configuration (sorted by error)', fontsize=12)
        plt.ylabel('Euclidean Error (cm)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'gravity_position_error.png'), dpi=300)
        plt.show()
        
        # 2. Joint deflection due to gravity
        plt.figure(figsize=(12, 6))
        
        # Extract joint deflection data for visualization (first 4 joints only)
        joint_deflections = np.array([row['angle_deflection'][:4] for _, row in results_df.iterrows()])
        
        # Convert to positive values for visualization (deflection magnitude)
        joint_deflections = np.abs(joint_deflections)
        
        # Create boxplots for each joint
        plt.boxplot(joint_deflections, labels=['Base', 'Shoulder', 'Elbow', 'Wrist'])
        plt.title('Joint Angle Deflection Due to Gravity', fontsize=14, fontweight='bold')
        plt.ylabel('Deflection Magnitude (degrees)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add individual points for each configuration
        for i in range(joint_deflections.shape[1]):
            # Spread points horizontally for better visibility
            x = np.random.normal(i+1, 0.06, size=joint_deflections.shape[0])
            plt.scatter(x, joint_deflections[:, i], alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'joint_deflection.png'), dpi=300)
        plt.show()
        
        # 3. 3D visualization of arm positions (ideal vs gravity-affected)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract position data
        ideal_pos = np.array([list(row['ideal_position']) for _, row in results_df.iterrows()])
        gravity_pos = np.array([list(row['gravity_position']) for _, row in results_df.iterrows()])
        
        # Plot ideal positions
        ax.scatter(ideal_pos[:, 0], ideal_pos[:, 1], ideal_pos[:, 2], 
                  c='blue', marker='o', s=50, label='Ideal (Simulation)')
        
        # Plot gravity-affected positions
        ax.scatter(gravity_pos[:, 0], gravity_pos[:, 1], gravity_pos[:, 2], 
                  c='red', marker='x', s=50, label='With Gravity')
        
        # Draw lines connecting corresponding points
        for i in range(len(ideal_pos)):
            ax.plot([ideal_pos[i, 0], gravity_pos[i, 0]],
                   [ideal_pos[i, 1], gravity_pos[i, 1]],
                   [ideal_pos[i, 2], gravity_pos[i, 2]],
                   'k-', alpha=0.3)
        
        # Add axes labels and title
        ax.set_xlabel('X (cm)', fontsize=12)
        ax.set_ylabel('Y (cm)', fontsize=12)
        ax.set_zlabel('Z (cm)', fontsize=12)
        ax.set_title('Ideal vs Gravity-Affected End Effector Positions', fontsize=14, fontweight='bold')
        ax.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'gravity_position_comparison_3d.png'), dpi=300)
        plt.show()
        
        # 4. Detailed visualization of a single configuration (first one by default)
        self.visualize_specific_configuration(results_df.iloc[0], output_dir)
        
        # 5. Generate summary statistics
        summary = pd.DataFrame({
            'Metric': ['Mean Position Error', 'Max Position Error', 'Mean Z Error (Gravity Direction)', 
                      'Mean Shoulder Deflection', 'Mean Elbow Deflection'],
            'Value': [
                f"{results_df['euclidean_error'].mean():.2f} cm",
                f"{results_df['euclidean_error'].max():.2f} cm",
                f"{results_df['z_error'].mean():.2f} cm",
                f"{np.mean(np.abs(joint_deflections[:, 1])):.2f}°",
                f"{np.mean(np.abs(joint_deflections[:, 2])):.2f}°"
            ]
        })
        
        print("\n======= Gravity Effect Analysis =======")
        print(summary.to_string(index=False))
        
        if output_dir:
            summary.to_csv(os.path.join(output_dir, 'gravity_effect_summary.csv'), index=False)
            results_df.to_csv(os.path.join(output_dir, 'detailed_gravity_analysis.csv'), index=False)
        
        return summary
    
    def visualize_specific_configuration(self, config_row, output_dir=None):
        """
        Visualize a specific configuration showing both ideal and gravity-affected arm positions
        
        Args:
            config_row: Row from results DataFrame for a specific configuration
            output_dir: Directory to save visualization
        """
        # Extract data for this configuration
        joint_angles = config_row['joint_angles']
        gravity_angles = config_row['gravity_affected_angles']
        
        # Get full joint positions for both configurations
        fk_ideal = self.forward_kinematics(joint_angles)
        fk_gravity = self.forward_kinematics(gravity_angles)
        
        ideal_positions = np.array(fk_ideal['positions'])
        gravity_positions = np.array(fk_gravity['positions'])
        
        # Create visualization with two perspectives
        fig = plt.figure(figsize=(16, 7))
        
        # Side view
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(elev=20, azim=30)
        
        # Top view
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=90, azim=-90)
        
        for ax in [ax1, ax2]:
            # Plot ideal arm configuration
            ax.plot(ideal_positions[:, 0], ideal_positions[:, 1], ideal_positions[:, 2], 
                   'bo-', linewidth=3, label='Ideal', alpha=0.7)
            
            # Plot gravity-affected configuration
            ax.plot(gravity_positions[:, 0], gravity_positions[:, 1], gravity_positions[:, 2], 
                   'ro-', linewidth=3, label='With Gravity', alpha=0.7)
            
            # Plot end effector positions
            ax.scatter([ideal_positions[-1, 0]], [ideal_positions[-1, 1]], [ideal_positions[-1, 2]], 
                      c='blue', marker='o', s=100)
            ax.scatter([gravity_positions[-1, 0]], [gravity_positions[-1, 1]], [gravity_positions[-1, 2]], 
                      c='red', marker='x', s=100)
            
            # Draw reference grid
            grid_size = max(30, max(np.max(ideal_positions), np.max(gravity_positions)) * 1.2)
            for line in range(-int(grid_size), int(grid_size) + 1, 5):
                ax.plot([-grid_size, grid_size], [line, line], [0, 0], 'lightgray', alpha=0.3)
                ax.plot([line, line], [-grid_size, grid_size], [0, 0], 'lightgray', alpha=0.3)
            
            # Set labels and legend
            ax.set_xlabel('X (cm)', fontsize=12)
            ax.set_ylabel('Y (cm)', fontsize=12)
            ax.set_zlabel('Z (cm)', fontsize=12)
            ax.legend()
        
        # Set title for the entire figure
        plt.suptitle(f'Configuration: {[round(a, 1) for a in joint_angles]}\n'
                   f'Error due to gravity: {config_row["euclidean_error"]:.2f} cm', 
                   fontsize=14, fontweight='bold')
        
        # Add joint deflection information
        deflection_text = "Joint Deflections due to Gravity:\n"
        for i in range(4):  # Include only the first 4 joints
            joint_names = ["Base", "Shoulder", "Elbow", "Wrist"]
            deflection = joint_angles[i] - gravity_angles[i]
            deflection_text += f"{joint_names[i]}: {deflection:.2f}°\n"
        
        ax1.text2D(0.05, 0.05, deflection_text, transform=ax1.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'specific_configuration.png'), dpi=300)
        plt.show()
        
        return fig
    
    def visualize_config_from_image(self, output_dir=None):
        """
        Visualize the specific configuration from the image [0, 0, 0, 0, 0, 67]
        with detailed gravity analysis
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Configuration from image
        image_config = [0, 0, 0, 0, 0, 67]
        
        # Calculate ideal position
        fk_ideal = self.forward_kinematics(image_config)
        ideal_position = fk_ideal['positions'][-1]
        
        # Calculate gravity-affected position
        gravity_angles = self.apply_gravity_effects(image_config)
        fk_gravity = self.forward_kinematics(gravity_angles)
        gravity_position = fk_gravity['positions'][-1]
        
        # Calculate error
        error_vector = np.array(gravity_position) - np.array(ideal_position)
        euclidean_error = np.linalg.norm(error_vector)
        
        # Create a result row similar to what's in the dataframe
        config_row = {
            'joint_angles': image_config,
            'gravity_affected_angles': gravity_angles,
            'ideal_position': ideal_position,
            'gravity_position': gravity_position,
            'euclidean_error': euclidean_error,
            'x_error': error_vector[0],
            'y_error': error_vector[1],
            'z_error': error_vector[2],
            'angle_deflection': [round(image_config[j] - gravity_angles[j], 2) for j in range(len(image_config))]
        }
        
        # Visualize this specific configuration
        fig = self.visualize_specific_configuration(config_row, output_dir)
        
        print("\n======= Analysis of Image Configuration =======")
        print(f"Joint angles: {[round(a, 1) for a in image_config]}")
        print(f"Gravity-affected angles: {[round(a, 1) for a in gravity_angles]}")
        print(f"Position error: {euclidean_error:.2f} cm")
        print(f"Error components: X={error_vector[0]:.2f}, Y={error_vector[1]:.2f}, Z={error_vector[2]:.2f}")
        
        # Joint deflections
        print("\nJoint deflections due to gravity:")
        joint_names = ["Base", "Shoulder", "Elbow", "Wrist", "Wrist rotation"]
        for i in range(5):
            deflection = image_config[i] - gravity_angles[i]
            print(f"{joint_names[i]}: {deflection:.2f}°")
        
        return config_row

def main():
    """Run gravity-aware error analysis."""
    # Create the robot with the same link lengths as in the main simulator
    robot = GravityAwareErrorAnalysis(link_lengths=[10, 15, 12, 8, 4])
    
    # Set up output directory
    output_dir = 'gravity_error_analysis'
    
    # First analyze the specific configuration from the image
    print("\n=== Analyzing the configuration from the image ===")
    image_config_results = robot.visualize_config_from_image(output_dir)
    
    # Then run the full error analysis
    print("\n=== Running complete gravity error analysis ===")
    results = robot.run_error_analysis()
    
    # Visualize the results
    print("\n=== Generating visualizations and summary ===")
    summary = robot.visualize_gravity_effects(results, output_dir)
    
    print("\n=== Analysis Complete ===")
    print(f"All results saved to: {output_dir}")
    
    # Determine if the gravity effects are significant enough to require compensation
    mean_error = results['euclidean_error'].mean()
    if mean_error > 0.5:  # If average error > 0.5cm
        print("\nRECOMMENDATION: Gravity compensation is needed for accurate positioning.")
        print("Options include:")
        print("1. Add counterweights to balance the arm")
        print("2. Increase joint stiffness (stronger servos or springs)")
        print("3. Implement software compensation by adjusting target positions")
    else:
        print("\nGravity effects are within acceptable limits for most applications.")

if __name__ == "__main__":
    main() 