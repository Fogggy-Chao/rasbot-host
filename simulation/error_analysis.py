import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import os
from pathlib import Path
from kinematics import RobotKinematics

class RobotErrorAnalysis(RobotKinematics):
    """
    Extends RobotKinematics to analyze error between simulation and real measurements.
    """
    def __init__(self, link_lengths=None, error_params=None):
        """
        Initialize with robot parameters and error characteristics.
        
        Args:
            link_lengths: List of link lengths for the robot
            error_params: Dict with error parameters (std_dev, bias, etc.)
        """
        super().__init__(link_lengths)
        
        # Default error parameters if not provided
        self.error_params = error_params or {
            'position_std_dev': 0.3,  # Standard deviation of position error (cm)
            'joint_std_dev': 0.8,     # Standard deviation of joint angle error (degrees)
            'bias': [0.1, 0.2, -0.15], # Systematic bias in x, y, z (cm)
            'nonlinear_factor': 0.05  # Factor for nonlinear distortion based on distance from origin
        }
    
    def generate_test_configurations(self, num_configs=30):
        """
        Generate a set of test joint configurations across the workspace.
        
        Args:
            num_configs: Number of configurations to generate
            
        Returns:
            List of joint angle configurations
        """
        configs = []
        
        # Generate configurations with good coverage of the workspace
        # Strategy: Mix systematic and random configurations
        
        # 1. Include important reference configurations
        reference_configs = [
            [90, 90, 90, 90, 90, 90],  # Home position
            [0, 0, 0, 0, 0, 90],        # Fully extended horizontal
            [0, 90, 90, 0, 0, 90],      # Vertical position
            [45, 45, 45, 45, 45, 90]    # Mid-range position
        ]
        configs.extend(reference_configs)
        
        # 2. Systematic configurations with varying single joint
        for joint_idx in range(5):  # For each joint (excluding gripper)
            for angle in [30, 60, 120, 150]:
                # Start with middle configuration
                config = [90, 90, 90, 90, 90, 90]
                # Vary one joint
                config[joint_idx] = angle
                configs.append(config)
        
        # 3. Random configurations for remaining slots
        remaining = num_configs - len(configs)
        for _ in range(remaining):
            random_config = []
            for limits in self.joint_limits:
                min_angle, max_angle = limits
                # Avoid extremes where physical robots might have issues
                margin = 10
                angle = np.random.uniform(min_angle + margin, max_angle - margin)
                random_config.append(angle)
            configs.append(random_config)
        
        return configs[:num_configs]  # Ensure we return exactly num_configs configurations
    
    def simulate_real_measurements(self, joint_angles):
        """
        Simulate "real" measurements by adding realistic error to the simulation.
        
        Args:
            joint_angles: List of joint angles in degrees
            
        Returns:
            Dict with 'real' position and the 'error' applied
        """
        # 1. Add error to joint angles to simulate encoder/motor errors
        joint_error = np.random.normal(0, self.error_params['joint_std_dev'], len(joint_angles))
        real_joint_angles = np.array(joint_angles) + joint_error
        
        # 2. Calculate end effector position with these modified angles
        real_fk = self.forward_kinematics(real_joint_angles)
        base_position = real_fk['positions'][-1]
        
        # 3. Add measurement noise to simulate sensor errors
        position_noise = np.random.normal(0, self.error_params['position_std_dev'], 3)
        
        # 4. Add systematic bias
        bias = np.array(self.error_params['bias'])
        
        # 5. Add nonlinear distortion based on distance from origin
        # (Simulates effects like link deflection under load)
        distance = np.linalg.norm(base_position)
        nonlinear_effect = np.array(base_position) * self.error_params['nonlinear_factor'] * distance / 100.0
        
        # Combine all error factors
        total_error = position_noise + bias + nonlinear_effect
        
        # Final "measured" position
        measured_position = np.array(base_position) + total_error
        
        return {
            'real_position': measured_position,
            'applied_error': total_error,
            'real_joint_angles': real_joint_angles,
            'joint_error': joint_error
        }
    
    def calculate_position_error(self, simulated_position, measured_position):
        """
        Calculate various error metrics between simulated and measured positions.
        
        Args:
            simulated_position: Position from simulation
            measured_position: Position from "real" measurement
            
        Returns:
            Dict with various error metrics
        """
        # Convert to numpy arrays
        sim_pos = np.array(simulated_position)
        meas_pos = np.array(measured_position)
        
        # Calculate error vector
        error_vector = meas_pos - sim_pos
        
        # Euclidean distance (overall error)
        euclidean_error = np.linalg.norm(error_vector)
        
        # Component-wise errors
        component_errors = np.abs(error_vector)
        
        return {
            'euclidean_error': euclidean_error,
            'error_vector': error_vector,
            'component_errors': component_errors,
            'x_error': error_vector[0],
            'y_error': error_vector[1],
            'z_error': error_vector[2]
        }
    
    def run_error_analysis(self, num_configs=30):
        """
        Run a complete error analysis over multiple configurations.
        
        Args:
            num_configs: Number of configurations to test
            
        Returns:
            DataFrame with all error analysis results
        """
        # Generate test configurations
        configs = self.generate_test_configurations(num_configs)
        
        # Prepare results storage
        results = []
        
        # Analyze each configuration
        for i, config in enumerate(configs):
            print(f"Analyzing configuration {i+1}/{len(configs)}: {[round(a, 1) for a in config]}")
            
            # Calculate simulation end effector position
            fk = self.forward_kinematics(config)
            sim_position = fk['positions'][-1]
            
            # Simulate "real" measurement
            real_measurement = self.simulate_real_measurements(config)
            measured_position = real_measurement['real_position']
            
            # Calculate error
            error_metrics = self.calculate_position_error(sim_position, measured_position)
            
            # Store results
            results.append({
                'config_id': i,
                'joint_angles': config,
                'simulated_position': sim_position,
                'measured_position': measured_position,
                'euclidean_error': error_metrics['euclidean_error'],
                'x_error': error_metrics['x_error'],
                'y_error': error_metrics['y_error'],
                'z_error': error_metrics['z_error'],
                'joint_error': real_measurement['joint_error'].tolist()
            })
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def visualize_error_distribution(self, results_df, output_dir=None):
        """
        Create visualizations of error distributions.
        
        Args:
            results_df: DataFrame with error analysis results
            output_dir: Directory to save visualizations
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Error histogram - overall Euclidean error
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['euclidean_error'], bins=15, alpha=0.8, color='#3498db')
        plt.axvline(results_df['euclidean_error'].mean(), color='r', linestyle='--', 
                    label=f'Mean: {results_df["euclidean_error"].mean():.3f}cm')
        plt.title('Distribution of Position Error', fontsize=14, fontweight='bold')
        plt.xlabel('Euclidean Error (cm)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'error_histogram.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Error by component (X, Y, Z)
        plt.figure(figsize=(10, 6))
        component_data = [
            results_df['x_error'],
            results_df['y_error'],
            results_df['z_error']
        ]
        plt.boxplot(component_data, labels=['X', 'Y', 'Z'], patch_artist=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.8),
                   medianprops=dict(color='red'))
        plt.title('Position Error by Component', fontsize=14, fontweight='bold')
        plt.ylabel('Error (cm)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'component_errors.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 3D Visualization of simulated vs measured positions
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions
        sim_positions = np.array([list(row['simulated_position']) for _, row in results_df.iterrows()])
        meas_positions = np.array([list(row['measured_position']) for _, row in results_df.iterrows()])
        
        # Plot simulated positions
        ax.scatter(sim_positions[:, 0], sim_positions[:, 1], sim_positions[:, 2], 
                  c='blue', marker='o', s=50, label='Simulated')
        
        # Plot measured positions
        ax.scatter(meas_positions[:, 0], meas_positions[:, 1], meas_positions[:, 2], 
                  c='red', marker='x', s=50, label='Measured')
        
        # Draw error lines connecting corresponding points
        for i in range(len(sim_positions)):
            ax.plot([sim_positions[i, 0], meas_positions[i, 0]],
                   [sim_positions[i, 1], meas_positions[i, 1]],
                   [sim_positions[i, 2], meas_positions[i, 2]],
                   'k-', alpha=0.3)
        
        # Add axes labels and title
        ax.set_xlabel('X (cm)', fontsize=12)
        ax.set_ylabel('Y (cm)', fontsize=12)
        ax.set_zlabel('Z (cm)', fontsize=12)
        ax.set_title('Simulated vs Measured Positions', fontsize=14, fontweight='bold')
        ax.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'position_comparison_3d.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Heat map of error vs position (projected on XY plane)
        plt.figure(figsize=(10, 8))
        
        # Create color map based on error magnitude
        plt.scatter(
            sim_positions[:, 0], sim_positions[:, 1],
            c=results_df['euclidean_error'], cmap='viridis',
            s=100, alpha=0.8, edgecolors='k'
        )
        
        plt.colorbar(label='Error (cm)')
        plt.title('Error Magnitude vs Position (XY Plane)', fontsize=14, fontweight='bold')
        plt.xlabel('X (cm)', fontsize=12)
        plt.ylabel('Y (cm)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'error_heatmap_xy.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Correlation between distance from origin and error
        plt.figure(figsize=(10, 6))
        
        # Calculate distance from origin for each simulated position
        distances = np.sqrt(np.sum(sim_positions**2, axis=1))
        
        plt.scatter(distances, results_df['euclidean_error'], alpha=0.8, c='#3498db', edgecolors='k')
        
        # Add trend line
        z = np.polyfit(distances, results_df['euclidean_error'], 1)
        p = np.poly1d(z)
        plt.plot(distances, p(distances), "r--", alpha=0.8,
                label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
        
        plt.title('Error vs Distance from Origin', fontsize=14, fontweight='bold')
        plt.xlabel('Distance from Origin (cm)', fontsize=12)
        plt.ylabel('Euclidean Error (cm)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'error_vs_distance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. Error analysis table
        summary = pd.DataFrame({
            'Metric': ['Mean Error', 'Median Error', 'Max Error', 'Min Error', 'Std Dev',
                      'Mean X Error', 'Mean Y Error', 'Mean Z Error'],
            'Value (cm)': [
                results_df['euclidean_error'].mean(),
                results_df['euclidean_error'].median(),
                results_df['euclidean_error'].max(),
                results_df['euclidean_error'].min(),
                results_df['euclidean_error'].std(),
                results_df['x_error'].abs().mean(),
                results_df['y_error'].abs().mean(),
                results_df['z_error'].abs().mean()
            ]
        })
        
        print("\n======= Error Analysis Summary =======")
        print(summary.to_string(index=False, float_format='%.3f'))
        
        if output_dir:
            summary.to_csv(os.path.join(output_dir, 'error_summary.csv'), index=False)
            
            # Also save the full results
            results_df.to_csv(os.path.join(output_dir, 'full_error_analysis.csv'), index=False)
        
        return summary

def main():
    """Run the robot error analysis with defined parameters."""
    # Create the error analysis object with the same link lengths as in the main simulation
    robot = RobotErrorAnalysis(link_lengths=[10, 15, 12, 8, 4])
    
    # Set up error parameters based on typical robot arm characteristics
    robot.error_params = {
        'position_std_dev': 0.25,   # Std dev of position error (cm)
        'joint_std_dev': 0.7,       # Std dev of joint angle error (degrees)
        'bias': [0.12, -0.08, 0.15], # Systematic bias in x, y, z (cm)
        'nonlinear_factor': 0.03    # Nonlinear factor
    }
    
    # Run the analysis
    print("\nRunning error analysis...")
    results = robot.run_error_analysis(num_configs=40)
    
    # Create output directory
    output_dir = 'error_analysis_results'
    
    # Visualize results
    print("\nGenerating visualizations...")
    summary = robot.visualize_error_distribution(results, output_dir)
    
    # Calculate whether the error is acceptable
    mean_error = results['euclidean_error'].mean()
    max_error = results['euclidean_error'].max()
    
    print("\n===== Error Acceptability Analysis =====")
    print(f"Mean Error: {mean_error:.3f} cm")
    print(f"Max Error: {max_error:.3f} cm")
    
    # Define acceptability thresholds
    acceptable_mean = 0.5  # cm
    acceptable_max = 1.0   # cm
    
    if mean_error <= acceptable_mean and max_error <= acceptable_max:
        print("\nResults: The simulation is within ACCEPTABLE error ranges!")
    elif mean_error <= acceptable_mean * 1.5 and max_error <= acceptable_max * 1.5:
        print("\nResults: The simulation has MARGINAL error ranges - may be acceptable for non-precision tasks.")
    else:
        print("\nResults: The simulation has UNACCEPTABLE error ranges - calibration is needed.")
    
    print("\nRecommendations:")
    if results['x_error'].abs().mean() > 0.3:
        print("- X-axis calibration needed")
    if results['y_error'].abs().mean() > 0.3:
        print("- Y-axis calibration needed")
    if results['z_error'].abs().mean() > 0.3:
        print("- Z-axis calibration needed")
        
    # Generate a calibration report if errors are high
    if mean_error > acceptable_mean or max_error > acceptable_max:
        print("\nGenerating calibration suggestions...")
        
        # Extract joint configurations with highest errors
        problem_configs = results.nlargest(5, 'euclidean_error')
        
        print("\nJoint configurations with highest errors:")
        for i, row in problem_configs.iterrows():
            print(f"Config {row['config_id']}: Error = {row['euclidean_error']:.3f} cm")
            print(f"  Joint angles: {[round(a, 1) for a in row['joint_angles']]}")
            print(f"  End effector position (sim): {[round(p, 2) for p in row['simulated_position']]}")
            print(f"  End effector position (real): {[round(p, 2) for p in row['measured_position']]}")
            print(f"  Error components: X={row['x_error']:.2f}, Y={row['y_error']:.2f}, Z={row['z_error']:.2f}")
            print()

if __name__ == "__main__":
    main() 