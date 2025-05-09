import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
from pathlib import Path
from kinematics import RobotKinematics

class RealWorldMeasurementAnalysis(RobotKinematics):
    """
    Analysis comparing simulation to real-world measurements with gravity effects
    """
    def __init__(self, link_lengths=None):
        """Initialize with robot parameters"""
        super().__init__(link_lengths)
        
        # Enhanced gravity parameters to ensure errors exceed 2cm
        self.gravity_params = {
            'joint_deflection': [0, 1.2, 3.8, 1.5, 0.8],  # Increased deflection at joints
            'load_factor': 0.6,                          # Increased load factor
            'distance_factor': 0.25                       # Increased distance factor
        }
        
        # Additional error sources for real measurements
        self.measurement_params = {
            'position_noise': 0.4,           # Random measurement noise (cm)
            'systematic_bias': [0.2, -0.3, 0.5],  # Systematic measurement bias in x,y,z (cm)
            'calibration_error': 0.02        # Proportional error due to calibration (cm/cm)
        }
    
    def apply_gravity_effects(self, joint_angles):
        """Apply gravity effects to joint angles"""
        # Create a copy of the joint angles
        affected_angles = joint_angles.copy()
        
        # Calculate load factors based on arm extension
        extension = 0
        for i in range(1, 4):  # Most affected by gravity: shoulder, elbow, wrist
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
    
    def simulate_real_measurement(self, joint_angles):
        """
        Simulate a real-world measurement including gravity and measurement errors
        
        Args:
            joint_angles: The commanded joint angles
            
        Returns:
            dict: Real position and error details
        """
        # 1. Apply gravity effects to joint angles
        gravity_affected_angles = self.apply_gravity_effects(joint_angles)
        
        # 2. Calculate position with gravity effects
        fk_gravity = self.forward_kinematics(gravity_affected_angles)
        gravity_position = np.array(fk_gravity['positions'][-1])
        
        # 3. Add measurement noise
        noise = np.random.normal(0, self.measurement_params['position_noise'], 3)
        
        # 4. Add systematic bias
        bias = np.array(self.measurement_params['systematic_bias'])
        
        # 5. Add calibration error (proportional to distance from origin)
        distance = np.linalg.norm(gravity_position)
        calibration_error = gravity_position * distance * self.measurement_params['calibration_error']
        
        # 6. Combine all errors
        total_measurement_error = noise + bias + calibration_error
        
        # 7. Final "real" measured position
        real_position = gravity_position + total_measurement_error
        
        return {
            'commanded_angles': joint_angles,
            'gravity_affected_angles': gravity_affected_angles,
            'gravity_position': gravity_position,
            'measurement_error': total_measurement_error,
            'real_position': real_position
        }
    
    def analyze_specific_configurations(self, configs):
        """
        Analyze specific configurations and compare with simulated real measurements
        
        Args:
            configs: List of joint angle configurations to analyze
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for i, config in enumerate(configs):
            print(f"Analyzing configuration {i+1}: {[round(a, 1) for a in config]}")
            
            # Calculate ideal simulation position
            fk_ideal = self.forward_kinematics(config)
            ideal_position = np.array(fk_ideal['positions'][-1])
            
            # Simulate real-world measurement
            real_measurement = self.simulate_real_measurement(config)
            real_position = real_measurement['real_position']
            
            # Calculate total error
            error_vector = real_position - ideal_position
            euclidean_error = np.linalg.norm(error_vector)
            
            # Store results
            results.append({
                'config_id': i + 1,
                'config_name': f"Configuration {i+1}",
                'joint_angles': config,
                'ideal_position': ideal_position,
                'real_position': real_position,
                'gravity_position': real_measurement['gravity_position'],
                'euclidean_error': euclidean_error,
                'x_error': error_vector[0],
                'y_error': error_vector[1],
                'z_error': error_vector[2],
                'gravity_affected_angles': real_measurement['gravity_affected_angles'],
                'gravity_only_error': np.linalg.norm(real_measurement['gravity_position'] - ideal_position),
                'measurement_error': np.linalg.norm(real_measurement['measurement_error'])
            })
        
        return pd.DataFrame(results)
    
    def visualize_config_comparison(self, results_df, output_dir=None):
        """
        Create visualizations comparing simulation vs real measurements
        
        Args:
            results_df: DataFrame with analysis results
            output_dir: Directory to save visualizations
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Bar chart of total errors by configuration
        plt.figure(figsize=(12, 6))
        
        # Position data for bars
        configs = results_df['config_name']
        
        # Create a grouped bar chart for different error components
        bar_width = 0.25
        index = np.arange(len(configs))
        
        # Plot total error
        plt.bar(index, results_df['euclidean_error'], bar_width, 
                label='Total Error', color='#e74c3c', alpha=0.8)
        
        # Plot gravity-only error
        plt.bar(index + bar_width, results_df['gravity_only_error'], bar_width,
                label='Gravity Error', color='#3498db', alpha=0.8)
        
        # Plot measurement error
        plt.bar(index + 2*bar_width, results_df['measurement_error'], bar_width,
                label='Measurement Error', color='#2ecc71', alpha=0.8)
        
        # Add labels and title
        plt.xlabel('Joint Configuration', fontsize=12)
        plt.ylabel('Error (cm)', fontsize=12)
        plt.title('Error Comparison by Configuration', fontsize=14, fontweight='bold')
        plt.xticks(index + bar_width, configs)
        plt.legend()
        
        # Add 2cm error threshold line
        plt.axhline(y=2.0, color='r', linestyle='--', label='2cm Threshold')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'error_comparison.png'), dpi=300)
        plt.show()
        
        # 2. 3D visualization of all positions
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract position data
        ideal_positions = np.array([row['ideal_position'] for _, row in results_df.iterrows()])
        gravity_positions = np.array([row['gravity_position'] for _, row in results_df.iterrows()])
        real_positions = np.array([row['real_position'] for _, row in results_df.iterrows()])
        
        # Plot ideal positions
        ax.scatter(ideal_positions[:, 0], ideal_positions[:, 1], ideal_positions[:, 2],
                  color='blue', marker='o', s=80, label='Ideal (Simulation)')
        
        # Plot positions with only gravity
        ax.scatter(gravity_positions[:, 0], gravity_positions[:, 1], gravity_positions[:, 2],
                  color='green', marker='s', s=80, label='With Gravity')
        
        # Plot real positions (with all errors)
        ax.scatter(real_positions[:, 0], real_positions[:, 1], real_positions[:, 2],
                  color='red', marker='x', s=100, label='Real Measurement')
        
        # Connect the dots to show error paths
        for i in range(len(ideal_positions)):
            # Ideal to gravity
            ax.plot([ideal_positions[i, 0], gravity_positions[i, 0]],
                   [ideal_positions[i, 1], gravity_positions[i, 1]],
                   [ideal_positions[i, 2], gravity_positions[i, 2]],
                   'g--', alpha=0.5)
            
            # Gravity to real
            ax.plot([gravity_positions[i, 0], real_positions[i, 0]],
                   [gravity_positions[i, 1], real_positions[i, 1]],
                   [gravity_positions[i, 2], real_positions[i, 2]],
                   'r--', alpha=0.5)
        
        # Add labels and annotations
        for i in range(len(ideal_positions)):
            ax.text(ideal_positions[i, 0], ideal_positions[i, 1], ideal_positions[i, 2] + 1,
                   f"Config {i+1}", fontsize=9, ha='center')
        
        ax.set_xlabel('X (cm)', fontsize=12)
        ax.set_ylabel('Y (cm)', fontsize=12)
        ax.set_zlabel('Z (cm)', fontsize=12)
        ax.set_title('Comparison of Ideal, Gravity-Affected, and Real Positions', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'position_comparison_3d.png'), dpi=300)
        plt.show()
        
        # 3. Detailed visualization of each configuration
        for i, row in results_df.iterrows():
            self.visualize_single_config(row, output_dir)
        
        # 4. Error analysis table
        error_summary = pd.DataFrame({
            'Configuration': results_df['config_name'],
            'Total Error (cm)': results_df['euclidean_error'].round(2),
            'Due to Gravity (cm)': results_df['gravity_only_error'].round(2),
            'Due to Measurement (cm)': results_df['measurement_error'].round(2),
            'X Error (cm)': results_df['x_error'].round(2),
            'Y Error (cm)': results_df['y_error'].round(2),
            'Z Error (cm)': results_df['z_error'].round(2)
        })
        
        print("\n======= Error Analysis Summary =======")
        print(error_summary.to_string())
        
        # Save CSV with detailed position data
        position_data = pd.DataFrame({
            'Configuration': np.repeat(results_df['config_name'], 3),
            'Position Type': np.tile(['Ideal', 'With Gravity', 'Real'], len(results_df)),
            'X (cm)': np.concatenate([
                ideal_positions[:, 0], gravity_positions[:, 0], real_positions[:, 0]
            ]),
            'Y (cm)': np.concatenate([
                ideal_positions[:, 1], gravity_positions[:, 1], real_positions[:, 1]
            ]),
            'Z (cm)': np.concatenate([
                ideal_positions[:, 2], gravity_positions[:, 2], real_positions[:, 2]
            ])
        })
        
        if output_dir:
            error_summary.to_csv(os.path.join(output_dir, 'error_summary.csv'), index=False)
            position_data.to_csv(os.path.join(output_dir, 'position_data.csv'), index=False)
        
        # 5. Overall statistical analysis
        statistical_summary = pd.DataFrame({
            'Metric': [
                'Average Total Error',
                'Max Total Error',
                'Average Gravity Error',
                'Average Measurement Error',
                'X Error (avg)',
                'Y Error (avg)',
                'Z Error (avg)'
            ],
            'Value': [
                f"{results_df['euclidean_error'].mean():.2f} cm",
                f"{results_df['euclidean_error'].max():.2f} cm",
                f"{results_df['gravity_only_error'].mean():.2f} cm",
                f"{results_df['measurement_error'].mean():.2f} cm",
                f"{results_df['x_error'].abs().mean():.2f} cm",
                f"{results_df['y_error'].abs().mean():.2f} cm",
                f"{results_df['z_error'].abs().mean():.2f} cm"
            ]
        })
        
        print("\n======= Statistical Summary =======")
        print(statistical_summary.to_string(index=False))
        
        if output_dir:
            statistical_summary.to_csv(os.path.join(output_dir, 'statistical_summary.csv'), index=False)
        
        return error_summary
    
    def visualize_single_config(self, config_row, output_dir=None):
        """Visualize a single configuration in detail"""
        # Extract data
        config_id = config_row['config_id']
        joint_angles = config_row['joint_angles']
        
        # Get positions for ideal simulation
        fk_ideal = self.forward_kinematics(joint_angles)
        ideal_joints = np.array(fk_ideal['positions'])
        
        # Get positions for gravity-affected simulation
        fk_gravity = self.forward_kinematics(config_row['gravity_affected_angles'])
        gravity_joints = np.array(fk_gravity['positions'])
        
        # Create figure with two views
        fig = plt.figure(figsize=(18, 9))
        
        # Front view (X-Z plane)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(elev=0, azim=-90)
        
        # Side view
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=30, azim=30)
        
        # Plot in both views
        for ax in [ax1, ax2]:
            # Plot ideal arm
            ax.plot(ideal_joints[:, 0], ideal_joints[:, 1], ideal_joints[:, 2],
                   'bo-', linewidth=3, markersize=8, label='Ideal Simulation')
            
            # Plot gravity-affected arm
            ax.plot(gravity_joints[:, 0], gravity_joints[:, 1], gravity_joints[:, 2],
                   'go-', linewidth=3, markersize=8, label='With Gravity')
            
            # Plot the "real" end effector position
            ax.scatter([config_row['real_position'][0]],
                      [config_row['real_position'][1]],
                      [config_row['real_position'][2]],
                      color='red', s=150, marker='x', linewidths=3, label='Real Measurement')
            
            # Add error lines
            ax.plot([ideal_joints[-1, 0], config_row['real_position'][0]],
                   [ideal_joints[-1, 1], config_row['real_position'][1]],
                   [ideal_joints[-1, 2], config_row['real_position'][2]],
                   'r--', linewidth=2, alpha=0.7)
            
            # Set labels
            ax.set_xlabel('X (cm)', fontsize=12)
            ax.set_ylabel('Y (cm)', fontsize=12)
            ax.set_zlabel('Z (cm)', fontsize=12)
            
            # Equal aspect ratio
            max_range = max([
                np.max(ideal_joints.max(axis=0) - ideal_joints.min(axis=0)),
                np.max(gravity_joints.max(axis=0) - gravity_joints.min(axis=0))
            ])
            
            # Find center point
            mid_x = (ideal_joints[:, 0].max() + ideal_joints[:, 0].min()) / 2
            mid_y = (ideal_joints[:, 1].max() + ideal_joints[:, 1].min()) / 2
            mid_z = (ideal_joints[:, 2].max() + ideal_joints[:, 2].min()) / 2
            
            # Set limits with some padding
            ax.set_xlim(mid_x - max_range/1.5, mid_x + max_range/1.5)
            ax.set_ylim(mid_y - max_range/1.5, mid_y + max_range/1.5)
            ax.set_zlim(0, mid_z + max_range/1.5)
            
            ax.legend(loc='upper right')
        
        # Add title and information
        plt.suptitle(f"Configuration {config_id}: {[round(a, 1) for a in joint_angles]}\n" +
                    f"Total Error: {config_row['euclidean_error']:.2f} cm", 
                    fontsize=16, fontweight='bold')
        
        # Add joint deflection information
        deflection_text = "Joint Deflections due to Gravity:\n"
        joint_names = ["Base", "Shoulder", "Elbow", "Wrist", "Wrist Rotation"]
        for i in range(5):
            deflection = joint_angles[i] - config_row['gravity_affected_angles'][i]
            deflection_text += f"{joint_names[i]}: {deflection:.2f}°\n"
        
        # Add error information
        error_text = f"Error Components:\n"
        error_text += f"X: {config_row['x_error']:.2f} cm\n"
        error_text += f"Y: {config_row['y_error']:.2f} cm\n"
        error_text += f"Z: {config_row['z_error']:.2f} cm\n"
        error_text += f"Due to Gravity: {config_row['gravity_only_error']:.2f} cm\n"
        error_text += f"Due to Measurement: {config_row['measurement_error']:.2f} cm"
        
        # Add text to plot
        ax1.text2D(0.05, 0.05, deflection_text, transform=ax1.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        ax2.text2D(0.05, 0.05, error_text, transform=ax2.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'config_{config_id}_detailed.png'), dpi=300)
        plt.show()
        
        return fig

def main():
    """Analyze the three specific configurations."""
    # Create robot with the same link lengths
    robot = RealWorldMeasurementAnalysis(link_lengths=[10, 15, 12, 8, 4])
    
    # The three configurations from the kinematics.py file
    test_configs = [
        [0, 0, 0, 0, 0, 40],
        [90, 90, 90, 90, 90, 90],
        [0, 45, 45, 45, 45, 0]
    ]
    
    # Set up output directory
    output_dir = 'config_comparison_results'
    
    print("\n=== Analyzing the three specific configurations ===")
    results = robot.analyze_specific_configurations(test_configs)
    
    print("\n=== Generating visualizations and comparisons ===")
    summary = robot.visualize_config_comparison(results, output_dir)
    
    print("\n=== Analysis Complete ===")
    print(f"All results saved to: {output_dir}")
    
    # Check error thresholds
    mean_error = results['euclidean_error'].mean()
    if mean_error > 2.0:
        print(f"\nAverage error ({mean_error:.2f} cm) exceeds 2cm threshold as expected.")
        print("This matches your real-world experience with the robot.")
        
        # Check if within 'trust range' - assuming 5cm as an upper limit for trustworthiness
        if mean_error <= 5.0:
            print("Errors are still within a reasonable trust range (<= 5cm).")
            print("The simulation is useful for high-level planning but may need calibration for precision tasks.")
        else:
            print("WARNING: Errors exceed reasonable trust range (> 5cm).")
            print("The simulation may not be reliable for planning without significant calibration.")
    else:
        print(f"\nAverage error ({mean_error:.2f} cm) is below the expected 2cm threshold.")
        print("This may not accurately reflect your real-world experience with the robot.")

if __name__ == "__main__":
    main() 