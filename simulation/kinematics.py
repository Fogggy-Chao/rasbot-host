import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random
from tqdm import tqdm
from matplotlib import cm
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class RobotKinematics:
    """
    Class for analyzing and visualizing robot kinematics using DH parameters.
    """
    def __init__(self, link_lengths=None):
        """
        Initialize the robot kinematics model.
        
        Args:
            link_lengths (list): List of link lengths [l1, l2, l3, l4, l5]
                                l1: Base to shoulder length
                                l2: Shoulder to elbow length
                                l3: Elbow to wrist length
                                l4: Wrist to wrist rotation length
                                l5: Wrist rotation to end effector length
        """
        # Default link lengths if not provided
        self.link_lengths = link_lengths or [10, 15, 12, 8, 4]
        
        # Joint limits in degrees (min, max)
        self.joint_limits = [
            (0, 180),    # Base joint
            (0, 180),    # Shoulder joint
            (0, 180),    # Elbow joint
            (0, 180),    # Wrist joint
            (0, 180),    # Wrist rotation joint
            (40, 100)    # End effector (gripper)
        ]
        
        # Servo parameters
        self.pulse_width_range = (0.5, 2.5)  # milliseconds
        self.angle_range = (0, 180)          # degrees
        
    def deg_to_rad(self, deg):
        """Convert degrees to radians."""
        return deg * np.pi / 180.0
    
    def get_dh_parameters(self, joint_angles):
        """
        Calculate the DH parameters for the robot.
        
        Args:
            joint_angles (list): List of joint angles in degrees [θ1, θ2, θ3, θ4, θ5, θ6]
        
        Returns:
            list: List of DH parameters [(θ, d, a, α), ...]
        """
        # Convert angles to radians
        theta = [self.deg_to_rad(angle) for angle in joint_angles[:-1]]  # Exclude gripper
        
        # DH parameters for each joint: (theta, d, a, alpha)
        dh_params = [
            (theta[0], self.link_lengths[0], 0, np.pi/2),         # Base rotation (vertical axis)
            (theta[1], 0, self.link_lengths[1], 0),               # Shoulder joint
            (theta[2], 0, self.link_lengths[2], 0),               # Elbow joint
            (theta[3], 0, self.link_lengths[3], np.pi/2),         # Wrist pitch
            (theta[4], 0, 0, -np.pi/2),                           # Wrist roll
            (0, self.link_lengths[4], 0, 0)                       # End effector
        ]
        return dh_params
    
    def transform_matrix(self, theta, d, a, alpha):
        """
        Calculate the homogeneous transformation matrix for DH parameters.
        
        Args:
            theta, d, a, alpha: DH parameters
        
        Returns:
            np.array: 4x4 transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
    
    def forward_kinematics(self, joint_angles):
        """
        Calculate forward kinematics using the DH parameter approach.
        Adjusted to match physical robot where [90,90,90,90,90,90] is the straight-up position.
        
        Args:
            joint_angles (list): Joint angles in degrees
            
        Returns:
            dict: Dictionary containing:
                'positions': List of joint positions
                'transformations': List of transformation matrices
        """
        # Convert degrees to radians
        theta = np.radians(joint_angles)
        
        # Ensure we have at least 6 joint angles (pad with zeros if needed)
        while len(theta) < 6:
            theta = np.append(theta, 0)
        
        # Define DH parameters [alpha, a, d, theta_offset]
        # Adjusted theta_offset by -pi/2 to match physical robot
        dh_params = [
            [np.pi/2, 0, self.link_lengths[0], -np.pi/2],        # Base to shoulder
            [0, self.link_lengths[1], 0, 0],                     # Shoulder joint
            [0, self.link_lengths[2], 0, np.pi/2],              # Elbow joint
            [-np.pi/2, 0, 0, 0],                                  # Wrist joint
            [-np.pi/2, 0, self.link_lengths[3], -np.pi/2],       # Wrist rotation
            [0, self.link_lengths[4], 0, np.pi/2]               # End effector
        ]
        
        # Initialize matrices
        positions = [[0, 0, 0]]  # Base position
        transformations = [np.eye(4)]  # Identity matrix for base
        T = np.eye(4)  # Current transformation matrix
        
        # For each joint
        for i in range(len(dh_params)):
            # DH parameters
            alpha, a, d, theta_offset = dh_params[i]
            
            # Calculate transformation matrix using DH convention
            # Add a negative sign to theta[i] for joint 3 (elbow) to reverse direction
            if i == 2 or i == 3:  # Elbow joint (index 2), wrist joint (index 3)
                ct = np.cos(-theta[i] + theta_offset)  # Note the negative sign
                st = np.sin(-theta[i] + theta_offset)  # Note the negative sign
            else:
                ct = np.cos(theta[i] + theta_offset)
                st = np.sin(theta[i] + theta_offset)
            
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            
            # DH transformation matrix
            A = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])
            
            # Update transformation
            T = T @ A
            
            # Store position and transformation
            positions.append(T[:3, 3].tolist())
            transformations.append(T.copy())
        
        # Convert positions to numpy array for convenience
        positions = np.array(positions)
        
        return {'positions': positions, 'transformations': transformations}

    def calculate_jacobian(self, joint_angles):
        """
        Calculate the Jacobian matrix at the given joint configuration.
        
        Args:
            joint_angles (list): List of joint angles in degrees
            
        Returns:
            np.array: 6xN Jacobian matrix (N = number of joints)
        """
        # Convert to radians
        theta = [self.deg_to_rad(angle) for angle in joint_angles[:-1]]
        
        # Get transformation matrices
        fk = self.forward_kinematics(joint_angles)
        T = fk['transformations']
        
        # End effector position
        end_pos = fk['positions'][-1]
        
        # Initialize Jacobian (6x5 for position and orientation)
        J = np.zeros((6, 5))
        
        # For each joint (excluding gripper)
        for i in range(5):
            # Joint axis (z-axis of the joint frame)
            z_i = T[i][:3, 2]
            
            # Joint position
            p_i = T[i][:3, 3]
            
            # Linear velocity component (cross product of joint axis and distance to end effector)
            J[:3, i] = np.cross(z_i, np.array(end_pos) - p_i)
            
            # Angular velocity component
            J[3:, i] = z_i
            
        return J

    def sample_joint_configuration(self):
        """
        Sample a random joint configuration within the joint limits.
        
        Returns:
            list: Random joint angles in degrees
        """
        joint_angles = []
        for i, (min_angle, max_angle) in enumerate(self.joint_limits):
            joint_angles.append(random.uniform(min_angle, max_angle))
        return joint_angles
    
    def calculate_reachable_workspace(self, num_samples=10000):
        """
        Calculate the reachable workspace using Monte Carlo sampling.
        
        Args:
            num_samples (int): Number of random samples to use
            
        Returns:
            list: List of reachable positions
        """
        reachable_positions = []
        
        print(f"Sampling {num_samples} configurations...")
        for _ in tqdm(range(num_samples)):
            # Sample random joint configuration
            joint_angles = self.sample_joint_configuration()
            
            # Calculate forward kinematics
            fk = self.forward_kinematics(joint_angles)
            
            # Get end effector position
            end_effector_position = fk['positions'][-1]
            
            # Add to reachable positions
            reachable_positions.append(end_effector_position)
        
        return reachable_positions
    
    def calculate_dexterous_workspace(self, num_samples=2000, orientation_threshold=30):
        """
        Calculate the dexterous workspace (positions reachable with multiple orientations).
        
        Args:
            num_samples (int): Number of position samples to test
            orientation_threshold (float): Threshold for orientation diversity in degrees
            
        Returns:
            list: List of dexterous positions
        """
        position_orientations = {}
        
        print(f"Testing {num_samples} positions for orientation diversity...")
        for _ in tqdm(range(num_samples)):
            # Sample random joint configuration
            joint_angles = self.sample_joint_configuration()
            
            # Calculate forward kinematics
            fk = self.forward_kinematics(joint_angles)
            
            # Get end effector position and orientation
            end_effector_position = fk['positions'][-1]
            end_effector_rotation = fk['transformations'][-1][:3, :3]  # Rotation matrix
            
            # Convert rotation to Euler angles
            roll, pitch, yaw = self.rotation_matrix_to_euler(end_effector_rotation)
            orientation = np.array([roll, pitch, yaw])
            
            # Convert position to tuple for dictionary key (with reduced precision)
            pos_tuple = tuple(np.round(end_effector_position, 1))  # Reduced precision for grouping
            
            # Track orientations for each position
            if pos_tuple in position_orientations:
                position_orientations[pos_tuple].append(orientation)
            else:
                position_orientations[pos_tuple] = [orientation]
        
        # Find positions with diverse orientations
        dexterous_positions = []
        
        for pos_tuple, orientations in position_orientations.items():
            if len(orientations) >= 3:  # Need at least 3 different orientations
                # Check if orientations are diverse enough
                orientation_diversity = False
                
                # Simplified diversity check
                for i in range(min(len(orientations), 5)):  # Check only up to 5 orientations
                    for j in range(i+1, min(len(orientations), 5)):
                        # Calculate angular difference between orientations
                        angular_diff = np.linalg.norm(np.array(orientations[i]) - np.array(orientations[j]))
                        
                        if np.rad2deg(angular_diff) > orientation_threshold:
                            orientation_diversity = True
                            break
                    
                    if orientation_diversity:
                        break
                
                if orientation_diversity:
                    dexterous_positions.append(np.array(pos_tuple))
        
        return dexterous_positions

    def plot_cylinder(self, ax, x_center, y_center, z_center, radius, height, color='gray', alpha=0.7, angle=0):
        """
        Plot a 3D cylinder to represent a joint or link.
        
        Args:
            ax: Matplotlib axis
            x_center, y_center, z_center: Center coordinates
            radius: Cylinder radius
            height: Cylinder height
            color: Cylinder color
            alpha: Transparency
            angle: Rotation angle in degrees
        """
        # Generate cylinder data
        z = np.linspace(z_center - height/2, z_center + height/2, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        # Calculate cylinder surface coordinates
        x_grid = radius * np.cos(theta_grid) + x_center
        y_grid = radius * np.sin(theta_grid) + y_center
        
        # Plot the cylinder
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=True)
        
        # Plot cylinder caps
        cap_z = [z_center - height/2, z_center + height/2]
        for z_cap in cap_z:
            x_cap = radius * np.cos(theta) + x_center
            y_cap = radius * np.sin(theta) + y_center
            z_cap_array = np.full_like(theta, z_cap)
            ax.plot(x_cap, y_cap, z_cap_array, color=color, alpha=alpha)
            
            # Fill the cap
            circle = Circle((x_center, y_center), radius, color=color, alpha=alpha)
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=z_cap)
    
    def plot_sphere(self, ax, x_center, y_center, z_center, radius, color='blue', alpha=0.7):
        """
        Plot a 3D sphere to represent a joint.
        
        Args:
            ax: Matplotlib axis
            x_center, y_center, z_center: Center coordinates
            radius: Sphere radius
            color: Sphere color
            alpha: Transparency
        """
        # Generate sphere data
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = radius * np.outer(np.cos(u), np.sin(v)) + x_center
        y = radius * np.outer(np.sin(u), np.sin(v)) + y_center
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center
        
        # Plot the sphere
        ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True)
    
    def plot_gripper(self, ax, pos, orientation, size=1.5, opened=True, color='silver'):
        """
        Plot a simplified gripper.
        
        Args:
            ax: Matplotlib axis
            pos: Position (x, y, z)
            orientation: Orientation matrix
            size: Size of the gripper
            opened: Whether the gripper is open or closed
            color: Gripper color
        """
        # Gripper dimensions
        width = size
        depth = size * 0.8
        height = size * 0.4
        
        # Gripper opening angle in radians (if opened)
        angle = 0.5 if opened else 0.1
        
        # Left finger
        left_pts = np.array([
            [-width/2, -depth/2, 0],
            [-width/2, depth/2, 0],
            [-width/2 - width*np.sin(angle), depth/2, height],
            [-width/2 - width*np.sin(angle), -depth/2, height]
        ])
        
        # Right finger
        right_pts = np.array([
            [width/2, -depth/2, 0],
            [width/2, depth/2, 0],
            [width/2 + width*np.sin(angle), depth/2, height],
            [width/2 + width*np.sin(angle), -depth/2, height]
        ])
        
        # Apply orientation and position
        for pts in [left_pts, right_pts]:
            for i in range(len(pts)):
                pts[i] = np.dot(orientation[:3,:3], pts[i]) + pos
        
        # Plot the fingers (connect points to form faces)
        for pts in [left_pts, right_pts]:
            # Bottom face
            ax.plot3D(
                [pts[0][0], pts[1][0], pts[2][0], pts[3][0], pts[0][0]], 
                [pts[0][1], pts[1][1], pts[2][1], pts[3][1], pts[0][1]], 
                [pts[0][2], pts[1][2], pts[2][2], pts[3][2], pts[0][2]], 
                color=color, alpha=0.8
            )
            
            # Connect bottom to top
            for i in range(4):
                ax.plot3D(
                    [pts[i][0], pts[(i+1)%4][0]], 
                    [pts[i][1], pts[(i+1)%4][1]], 
                    [pts[i][2], pts[(i+1)%4][2]], 
                    color=color, alpha=0.8
                )
    
    def visualize_robot_enhanced(self, joint_angles, ax_to_use=None, show_plot=True, title_override=None):
        """
        Visualize the robot with enhanced 3D representation.
        All links represented by cylinders and joints by small spheres.
        
        Args:
            joint_angles (list): List of joint angles in degrees
            ax_to_use (matplotlib.axes.Axes, optional): Existing 3D axes to plot on. Defaults to None (creates new figure).
            show_plot (bool, optional): Whether to call plt.show(). Defaults to True.
            title_override (str, optional): Custom title for the plot. Defaults to None.
        """
        # First calculate forward kinematics to get joint positions
        fk = self.forward_kinematics(joint_angles)
        positions = np.array(fk['positions'])
        
        if ax_to_use:
            ax = ax_to_use
            ax.cla() # Clear the existing axis for the new frame
            fig = ax.get_figure()
        else:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Define aesthetically pleasing color palette for joints and links
        joint_colors = ['#2C3E50', '#3498DB', '#9B59B6', '#2ECC71', '#F1C40F', '#E74C3C']
        link_colors = ['#34495E', '#2980B9', '#8E44AD', '#27AE60', '#F39C12', '#C0392B']
        
        # Joint sizes (smaller than before)
        joint_sizes = [0.7, 0.6, 0.6, 0.5, 0.5, 0.5]
        
        # Link sizes (radius of cylinders)
        num_links = len(positions) - 1
        link_sizes = [0.7, 0.5, 0.5, 0.4, 0.4, 0.4]
        
        # Ensure we have enough link sizes
        while len(link_sizes) < num_links:
            link_sizes.append(0.4)
            
        # Plot elegant ground plane
        base_size = 20
        x = np.linspace(-base_size, base_size, 2)
        y = np.linspace(-base_size, base_size, 2)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Create elegant base plate with subtle gradient
        ax.plot_surface(X, Y, Z, alpha=0.3, color='#ECF0F1', shade=True)
        
        # Add grid lines for reference
        ax.plot([-base_size, base_size], [0, 0], [0, 0], 'gray', alpha=0.3, linestyle='--')
        ax.plot([0, 0], [-base_size, base_size], [0, 0], 'gray', alpha=0.3, linestyle='--')
        
        # Print joint positions for debugging
        print("Joint positions:")
        for i, pos in enumerate(positions):
            print(f"Joint {i}: {pos}")
        
        # Draw base cylinder
        self.plot_cylinder(ax, 0, 0, positions[0, 2]/2, 
                         0.8, positions[0, 2], color='#7f8c8d', alpha=0.9)
        
        # Draw joints and links
        for i in range(len(positions)-1):
            # Current and next position
            p1 = positions[i]
            p2 = positions[i+1]
            
            # Draw joint as sphere
            joint_idx = min(i, len(joint_sizes)-1)  # Safety check
            self.plot_sphere(ax, p1[0], p1[1], p1[2], joint_sizes[joint_idx], 
                            color=joint_colors[joint_idx % len(joint_colors)], alpha=0.9)
            
            # Calculate link direction and length
            direction = p2 - p1
            length = np.linalg.norm(direction)
            
            if length > 0:
                # Calculate link midpoint
                midpoint = (p1 + p2) / 2
                
                # Get link radius
                link_idx = min(i, len(link_sizes)-1)
                radius = link_sizes[link_idx]
                
                # Draw a thick line to represent the link
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       color=link_colors[link_idx % len(link_colors)], 
                       linewidth=radius*15, alpha=0.8)
        
        # Draw end effector joint
        if len(positions) > 1:
            end_idx = min(len(positions)-1, len(joint_sizes)-1)
            self.plot_sphere(ax, positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                            joint_sizes[end_idx], 
                            color=joint_colors[end_idx % len(joint_colors)], alpha=0.9)
        
        ax.set_xlabel('X (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (cm)', fontsize=12, fontweight='bold')
        
        if title_override:
            ax.set_title(title_override, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Robot Configuration: {[round(a, 1) for a in joint_angles]}°', 
                        fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = np.max(positions.max(axis=0) - positions.min(axis=0)) / 2
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2
        ax.set_xlim(mid_x - max_range*1.5, mid_x + max_range*1.5)
        ax.set_ylim(mid_y - max_range*1.5, mid_y + max_range*1.5)
        ax.set_zlim(0, mid_z + max_range*2)
        
        plt.tight_layout()
        if show_plot and not ax_to_use: # Only show if creating a new plot or explicitly asked
            plt.show()
        
        return fig, ax
    
    def animate_robot(self, start_angles, end_angles, num_frames=50):
        """
        Create an animation of the robot moving from start to end configuration.
        
        Args:
            start_angles (list): Starting joint angles in degrees
            end_angles (list): Ending joint angles in degrees
            num_frames (int): Number of frames in the animation
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Interpolation between start and end angles
        angle_steps = []
        for i in range(len(start_angles)):
            angle_steps.append(np.linspace(start_angles[i], end_angles[i], num_frames))
        
        # Initialize plot with empty line
        line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=8)
        
        # Set equal aspect ratio
        fk_start = self.forward_kinematics(start_angles)
        fk_end = self.forward_kinematics(end_angles)
        positions_start = np.array(fk_start['positions'])
        positions_end = np.array(fk_end['positions'])
        
        # Combine positions for setting axis limits
        all_positions = np.vstack((positions_start, positions_end))
        max_range = np.max(all_positions.max(axis=0) - all_positions.min(axis=0)) / 2
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) / 2
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) / 2
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Robot Motion Animation')
        
        def update(frame):
            # Get joint angles for this frame
            current_angles = [angle_steps[i][frame] for i in range(len(start_angles))]
            
            # Calculate forward kinematics
            fk = self.forward_kinematics(current_angles)
            positions = np.array(fk['positions'])
            
            # Update line data
            line.set_data(positions[:, 0], positions[:, 1])
            line.set_3d_properties(positions[:, 2])
            
            # Update title
            ax.set_title(f'Frame {frame}: {[round(a, 1) for a in current_angles]}')
            
            return line,
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def visualize_workspace_enhanced(self, num_samples=50000):
        """
        Visualize the reachable and dexterous workspace with three views.
        The robot arm is not shown to focus on the workspace point cloud.
        
        Args:
            num_samples (int): Number of samples for Monte Carlo simulation
        """
        # Calculate workspaces
        print("Calculating reachable workspace...")
        reachable_workspace = self.calculate_reachable_workspace(num_samples)
        print(f"Found {len(reachable_workspace)} reachable points")
        
        print("Calculating dexterous workspace...")
        dexterous_workspace = self.calculate_dexterous_workspace(num_samples//5)
        print(f"Found {len(dexterous_workspace)} dexterous points")
        
        # Convert to numpy arrays
        reachable_workspace = np.array(reachable_workspace)
        dexterous_workspace = np.array(dexterous_workspace) if len(dexterous_workspace) > 0 else np.empty((0, 3))
        
        # Create figure with three subplots - modern design
        fig = plt.figure(figsize=(18, 6), facecolor='white')
        
        # Create subplots with light background
        ax1 = fig.add_subplot(131, projection='3d', facecolor='#F8F9F9')
        ax1.view_init(90, -90)  # Top-down view
        
        ax2 = fig.add_subplot(132, projection='3d', facecolor='#F8F9F9')
        ax2.view_init(20, -60)  # Side view
        
        ax3 = fig.add_subplot(133, projection='3d', facecolor='#F8F9F9')
        ax3.view_init(20, 30)  # Another side view
        
        axes = [ax1, ax2, ax3]
        
        # Elegant base dimensions
        base_size = 20
        
        for ax in axes:
            # Create minimalist grid on the ground plane
            x = np.linspace(-base_size, base_size, 11)
            y = np.linspace(-base_size, base_size, 11)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Plot subtle grid
            for i in range(len(x)):
                ax.plot([x[i], x[i]], [-base_size, base_size], [0, 0], 'gray', alpha=0.1)
                ax.plot([-base_size, base_size], [y[i], y[i]], [0, 0], 'gray', alpha=0.1)
            
            # Plot ground plane
            ax.plot_surface(X, Y, Z, alpha=0.1, color='#ECF0F1')
            
            # Mark origin with subtle marker
            ax.scatter([0], [0], [0], color='black', s=20, marker='+')
            
            # Plot the workspace points with improved aesthetics
            ax.scatter(reachable_workspace[:, 0], reachable_workspace[:, 1], reachable_workspace[:, 2], 
                      c='#3498DB', alpha=0.05, s=0.8, label=f'Reachable Workspace ({len(reachable_workspace)} points)')
            
            if len(dexterous_workspace) > 0:
                ax.scatter(dexterous_workspace[:, 0], dexterous_workspace[:, 1], dexterous_workspace[:, 2], 
                          c='#E74C3C', alpha=0.2, s=1.5, label='Dexterous Workspace')
        
        # Set labels and titles with elegant styling
        titles = ['Top View', 'Front View', 'Side View']
        
        for i, ax in enumerate(axes):
            ax.set_xlabel('X (cm)', fontsize=12, labelpad=10)
            ax.set_ylabel('Y (cm)', fontsize=12, labelpad=10)
            ax.set_zlabel('Z (cm)', fontsize=12, labelpad=10)
            ax.set_title(titles[i], fontsize=14, fontweight='bold', pad=10)
            
            # Only show legend on first plot with elegant styling
            if i == 0:
                legend = ax.legend(loc='upper right', framealpha=0.7, fontsize=10)
            
            # Set axis limits with some padding
            max_reach = max(np.max(np.abs(reachable_workspace)), 30)
            ax.set_xlim(-max_reach*1.1, max_reach*1.1)
            ax.set_ylim(-max_reach*1.1, max_reach*1.1)
            ax.set_zlim(-0.1, max_reach*1.2)
            
            # Remove grid lines for cleaner look
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')
                
        plt.tight_layout()
        plt.show()
        
        return fig, axes

    def rotation_matrix_to_euler(self, R):
        """
        Convert a rotation matrix to Euler angles (roll, pitch, yaw).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            tuple: (roll, pitch, yaw) angles in radians
        """
        # Check if pitch is close to +/- 90 degrees (gimbal lock case)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy > 1e-6:
            # Normal case
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock case
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
            
        return roll, pitch, yaw

    def visualize_robot_three_views(self, joint_angles):
        """
        Visualize the robot kinematics from three different views (top, front, side).
        End effector is represented as a plane attached directly to the wrist rotation joint.
        
        Args:
            joint_angles (list): List of joint angles in degrees
            
        Returns:
            tuple: (fig, axes) matplotlib figure and axes
        """
        # Calculate forward kinematics
        fk = self.forward_kinematics(joint_angles)
        positions = np.array(fk['positions'])
        transforms = fk['transformations']
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(18, 6))
        
        # Create three different views
        ax1 = fig.add_subplot(131, projection='3d')  # Top view
        ax1.view_init(90, -90)
        
        ax2 = fig.add_subplot(132, projection='3d')  # Front view
        ax2.view_init(0, -90)
        
        ax3 = fig.add_subplot(133, projection='3d')  # Side view
        ax3.view_init(0, 0)
        
        axes = [ax1, ax2, ax3]
        titles = ['Top View', 'Front View', 'Side View']
        
        # Define colors for joints and links
        joint_colors = ['#2C3E50', '#3498DB', '#9B59B6', '#2ECC71', '#F1C40F']
        link_colors = ['#34495E', '#2980B9', '#8E44AD', '#27AE60', '#F39C12']
        
        # Joint and link sizes
        joint_sizes = [0.7, 0.6, 0.6, 0.5, 0.5]
        num_links = 5  # We'll have 5 joints
        link_sizes = [0.7, 0.5, 0.5, 0.4, 0.4]
        
        # Get dimensions for setting consistent axis limits
        max_range = np.max(positions.max(axis=0) - positions.min(axis=0)) / 2
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2
        
        # Define end effector plane size
        ee_size = 1.5
        
        # Extract transform matrix for end effector orientation
        # Use the wrist rotation transform (index 5)
        final_transform = transforms[5]
        
        # Get wrist rotation position
        ee_pos = positions[5]
        
        # Extract rotation matrix
        R = final_transform[:3, :3]
        
        # Calculate the corners of a square in the local coordinate system
        # These define a plane perpendicular to the x-axis
        local_corners = [
            [0, ee_size, ee_size],    # Front right
            [0, ee_size, -ee_size],   # Front left
            [0, -ee_size, -ee_size],  # Back left
            [0, -ee_size, ee_size]    # Back right
        ]
        
        # Transform the corners to the global coordinate system
        global_corners = []
        for corner in local_corners:
            # Apply rotation
            rotated_corner = R @ np.array(corner)
            # Apply translation
            global_corner = ee_pos + rotated_corner
            global_corners.append(global_corner)
        
        # Convert to array for easier manipulation
        global_corners = np.array(global_corners)
        
        # Draw robot in each view
        for i, ax in enumerate(axes):
            # Set title and view
            ax.set_title(titles[i], fontsize=14, fontweight='bold')
            
            # Add grid lines
            base_size = max(20, max_range * 2)
            for line in range(-int(base_size), int(base_size) + 1, 5):
                ax.plot([-base_size, base_size], [line, line], [0, 0], 'lightgray', alpha=0.3, linestyle='-')
                ax.plot([line, line], [-base_size, base_size], [0, 0], 'lightgray', alpha=0.3, linestyle='-')
            
            # Draw horizontal reference plane
            x = np.linspace(-base_size, base_size, 2)
            y = np.linspace(-base_size, base_size, 2)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            ax.plot_surface(X, Y, Z, alpha=0.1, color='#ECF0F1')
            
            # Draw base cylinder
            self.plot_cylinder(ax, 0, 0, positions[0, 2]/2, 
                             0.8, positions[0, 2], color='#7f8c8d', alpha=0.9)
            
            # Draw joints and links up to wrist rotation (index 5)
            for j in range(5):  # Only draw up to wrist rotation
                # Current and next position
                p1 = positions[j]
                p2 = positions[j+1]
                
                # Draw joint as sphere
                joint_idx = min(j, len(joint_sizes)-1)
                self.plot_sphere(ax, p1[0], p1[1], p1[2], joint_sizes[joint_idx], 
                                color=joint_colors[joint_idx % len(joint_colors)], alpha=0.9)
                
                # Draw link as thick line
                link_idx = min(j, len(link_sizes)-1)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       color=link_colors[link_idx % len(link_colors)], 
                       linewidth=link_sizes[link_idx]*15, alpha=0.8)
            
            # Draw the wrist rotation joint (the last joint - index 5)
            wrist_rot_idx = 5
            self.plot_sphere(ax, positions[wrist_rot_idx, 0], positions[wrist_rot_idx, 1], positions[wrist_rot_idx, 2], 
                            joint_sizes[-1], color=joint_colors[-1], alpha=0.9)
            
            # Draw the end effector plane (attached directly to wrist rotation)
            plane = Poly3DCollection([global_corners], alpha=0.7, linewidth=1, edgecolor='k')
            plane.set_facecolor('#E74C3C')
            ax.add_collection3d(plane)
            
            # Set labels
            ax.set_xlabel('X (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (cm)', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z (cm)', fontsize=12, fontweight='bold')
            
            # Set equal aspect ratio with some padding
            ax.set_xlim(mid_x - max_range*1.5, mid_x + max_range*1.5)
            ax.set_ylim(mid_y - max_range*1.5, mid_y + max_range*1.5)
            ax.set_zlim(0, mid_z + max_range*2)
            
            # Add position information
            if i == 0:  # Only on the first plot
                text = "Joint Positions:\n"
                for j in range(6):  # Include only up to wrist rotation
                    pos = positions[j]
                    text += f"J{j}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                ax.text2D(0.05, 0.05, text, transform=ax.transAxes, fontsize=9, 
                          bbox=dict(facecolor='white', alpha=0.7))
            
            # Add metadata to each view
            ax.text2D(0.05, 0.95, f"Angles: {[round(a, 1) for a in joint_angles]}°", 
                     transform=ax.transAxes, fontsize=9, 
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes

    def inverse_kinematics(self, target_position_xyz, initial_guess_angles_deg=None, gripper_angle_deg=70.0, max_iterations=1000, tolerance=0.1):
        """
        Calculate inverse kinematics for the robot arm to reach a target position.
        Uses the Cyclic Coordinate Descent (CCD) algorithm.

        Args:
            target_position_xyz (list or np.array): Target [x, y, z] coordinates.
            initial_guess_angles_deg (list, optional): Initial guess for joint angles [J0-J4] in degrees.
                                                       Defaults to mid-range of joint limits.
            gripper_angle_deg (float): Angle for the gripper (J5) in degrees. Not solved by IK.
            max_iterations (int): Maximum number of iterations for CCD.
            tolerance (float): Tolerance in cm for reaching the target.

        Returns:
            list or None: List of 5 joint angles [J0-J4] in degrees if a solution is found,
                          None otherwise.
        """
        target_pos = np.array(target_position_xyz)

        # 1. Reachability Check (simplified)
        # The robot's base link (l1) is primarily a vertical offset.
        l1_base_height = self.link_lengths[0]
        # Effective arm length is the sum of lengths of segments that contribute to reach from the shoulder pivot.
        # These are l2, l3, l4, l5 in the user's description, which are self.link_lengths[1] to self.link_lengths[4].
        effective_arm_length = sum(self.link_lengths[1:])

        # Calculate distance from the shoulder pivot (approx. at (0,0,l1_base_height)) to the target.
        dist_to_target_from_shoulder_base = np.linalg.norm(
            np.array([target_pos[0], target_pos[1], target_pos[2] - l1_base_height])
        )

        if dist_to_target_from_shoulder_base > effective_arm_length:
            print(f"Target likely unreachable: distance from shoulder base {dist_to_target_from_shoulder_base:.2f} cm > max arm length {effective_arm_length:.2f} cm.")
            # A more sophisticated check might also consider minimum reachability (holes in workspace).
            return None

        # Initialize joint angles (J0 to J4 - 5 arm joints)
        if initial_guess_angles_deg is None:
            current_angles_deg = np.array([
                (limit[0] + limit[1]) / 2.0 for limit in self.joint_limits[:5]
            ])
        else:
            current_angles_deg = np.array(initial_guess_angles_deg[:5], dtype=float)

        num_arm_joints = 5

        for iteration in range(max_iterations):
            # Angles for FK need to include the fixed gripper angle
            fk_angles_deg = np.append(current_angles_deg, gripper_angle_deg)

            fk_result = self.forward_kinematics(fk_angles_deg)
            # joint_positions[0] is base origin, positions[1] is origin of J0, ..., positions[6] is EE tip
            joint_positions = fk_result['positions']
            # joint_transforms[0] is transform for J0, ..., joint_transforms[5] is for J5 (gripper)
            joint_transforms = fk_result['transformations']
            
            current_ee_pos = joint_positions[6] # End effector position

            # Check for convergence
            error = np.linalg.norm(target_pos - current_ee_pos)
            if error < tolerance:
                print(f"Converged in {iteration + 1} iterations. Error: {error:.4f} cm.")
                return current_angles_deg.tolist()

            # CCD: Iterate through each arm joint (e.g., from wrist roll J4 down to base J0)
            for j_idx in range(num_arm_joints - 1, -1, -1):
                # P_j: origin of joint j_idx. For joint j (0-4), its origin is joint_positions[j_idx]
                # axis_j: rotation axis of joint j_idx in world frame. This is Z-axis of T[j_idx]
                P_j = joint_positions[j_idx]
                axis_j_world = joint_transforms[j_idx][:3, 2]

                vec_joint_to_ee = current_ee_pos - P_j
                vec_joint_to_target = target_pos - P_j

                norm_vec_joint_to_ee = np.linalg.norm(vec_joint_to_ee)
                norm_vec_joint_to_target = np.linalg.norm(vec_joint_to_target)

                if norm_vec_joint_to_ee < 1e-5 or norm_vec_joint_to_target < 1e-5:
                    continue # Avoid issues with zero-length vectors

                vec_joint_to_ee_norm = vec_joint_to_ee / norm_vec_joint_to_ee
                vec_joint_to_target_norm = vec_joint_to_target / norm_vec_joint_to_target

                # Calculate signed angle to rotate vec_joint_to_ee_norm to vec_joint_to_target_norm around axis_j_world
                # angle = atan2(dot(axis, cross(v1, v2)), dot(v1, v2))
                # Here, v1 is vec_joint_to_ee_norm, v2 is vec_joint_to_target_norm, axis is axis_j_world
                dot_v1_v2 = np.dot(vec_joint_to_ee_norm, vec_joint_to_target_norm)
                cross_v1_v2 = np.cross(vec_joint_to_ee_norm, vec_joint_to_target_norm)
                
                # Ensure dot_v1_v2 is within [-1, 1] for stability with atan2's internal calculations if it uses acos
                dot_v1_v2 = np.clip(dot_v1_v2, -1.0, 1.0)

                signed_angle_delta_rad = np.arctan2(np.dot(axis_j_world, cross_v1_v2), dot_v1_v2)
                
                if np.abs(signed_angle_delta_rad) < 1e-5: # If change is too small
                    continue

                # Update joint angle
                current_angle_j_deg = current_angles_deg[j_idx]
                damping_factor = 0.5 # Introduce a damping factor
                angle_change_deg = damping_factor * np.rad2deg(signed_angle_delta_rad)
                
                # Account for negated theta in FK for specific joints
                # j_idx in CCD corresponds to joint index i in FK's dh_params and theta array
                if j_idx == 2 or j_idx == 3:
                    angle_change_deg = -angle_change_deg
                    
                updated_angle_j_deg = current_angle_j_deg + angle_change_deg

                # Clamp to joint limits
                min_lim_deg, max_lim_deg = self.joint_limits[j_idx]
                updated_angle_j_deg_clamped = np.clip(updated_angle_j_deg, min_lim_deg, max_lim_deg)
                
                # If clamped angle is same as current, no effective change, skip re-FK for this joint
                if np.isclose(current_angles_deg[j_idx], updated_angle_j_deg_clamped):
                    continue

                current_angles_deg[j_idx] = updated_angle_j_deg_clamped

                # Recompute FK for the next joint in this iteration or for convergence check
                fk_angles_deg_updated = np.append(current_angles_deg, gripper_angle_deg)
                fk_result_updated = self.forward_kinematics(fk_angles_deg_updated)
                
                current_ee_pos = fk_result_updated['positions'][6]
                joint_positions = fk_result_updated['positions'] # Update for subsequent joints in this iteration
                joint_transforms = fk_result_updated['transformations']

                # Check error again to potentially break early
                error = np.linalg.norm(target_pos - current_ee_pos)
                if error < tolerance:
                    break # Break from inner CCD joint loop
            
            if error < tolerance: # Check if inner loop broke due to convergence
                print(f"Converged in {iteration + 1} iterations. Error: {error:.4f} cm.")
                return current_angles_deg.tolist()

        print(f"IK failed to converge after {max_iterations} iterations. Final error: {error:.4f} cm.")
        return None

    def plan_trajectory(self, start_config_deg, end_target_xyz, num_waypoints=20, ik_max_iterations=1500, ik_tolerance=0.5):
        """
        Plans a trajectory from a start joint configuration to an end-effector target position.
        Uses linear interpolation in joint space (LISP) after solving IK for the end point.

        Args:
            start_config_deg (list): List of 6 starting joint angles [J0-J5] in degrees.
            end_target_xyz (list or np.array): Target [x, y, z] coordinates for the end-effector.
            num_waypoints (int): Number of waypoints in the trajectory (including start and end).
            ik_max_iterations (int): Max iterations for the IK solver for the end point.
            ik_tolerance (float): Tolerance for the IK solver for the end point.

        Returns:
            list or None: A list of trajectory waypoints (each waypoint is a list of 6 joint angles),
                          or None if the end target is unreachable or IK fails.
        """
        if len(start_config_deg) != 6:
            print("Error: start_config_deg must contain 6 joint angles (J0-J5).")
            return None
        
        start_angles_arm_deg = start_config_deg[:5]
        gripper_angle_deg = start_config_deg[5]

        print(f"Planning trajectory from start config: {[round(a, 2) for a in start_config_deg]}")
        print(f"To end-effector target XYZ: {[round(c, 2) for c in end_target_xyz]} with {num_waypoints} waypoints.")

        # 1. Solve IK for the end_target_xyz
        # First attempt: use start_angles_arm_deg as initial guess
        end_angles_arm_deg_ik = self.inverse_kinematics(
            end_target_xyz,
            initial_guess_angles_deg=start_angles_arm_deg, # Use start arm angles as IK guess
            gripper_angle_deg=gripper_angle_deg, # Pass through the gripper angle
            max_iterations=ik_max_iterations,
            tolerance=ik_tolerance
        )

        # Second attempt: if first failed, try with default initial guess (None)
        if end_angles_arm_deg_ik is None:
            print(f"IK with start_config guess failed for {end_target_xyz}. Retrying IK with default initial guess.")
            end_angles_arm_deg_ik = self.inverse_kinematics(
                end_target_xyz,
                initial_guess_angles_deg=None, # Use default (mid-range)
                gripper_angle_deg=gripper_angle_deg,
                max_iterations=ik_max_iterations,
                tolerance=ik_tolerance
            )

        if end_angles_arm_deg_ik is None:
            print(f"Trajectory planning failed: Could not find IK solution for end target {end_target_xyz} even with default guess.")
            return None
        
        print(f"IK solution for end target found: {[round(a, 2) for a in end_angles_arm_deg_ik]}")

        # 2. Linear interpolation in joint space for the 5 arm joints
        trajectory_waypoints = []
        joint_space_trajectory = np.zeros((num_waypoints, 5)) # For J0-J4

        for i in range(5): # For each of the 5 arm joints
            joint_space_trajectory[:, i] = np.linspace(
                start_angles_arm_deg[i],
                end_angles_arm_deg_ik[i],
                num_waypoints
            )
        
        # Assemble full waypoints including the constant gripper angle
        for i in range(num_waypoints):
            waypoint = joint_space_trajectory[i, :].tolist() + [gripper_angle_deg]
            trajectory_waypoints.append(waypoint)
            
        print(f"Trajectory planned successfully with {len(trajectory_waypoints)} waypoints.")
        return trajectory_waypoints

def plot_joint_space_trajectory(trajectory_waypoints, title="Joint Space Trajectory (LISP)"):
    """
    Plots the joint angles over waypoints to illustrate LISP.

    Args:
        trajectory_waypoints (list): List of waypoints, where each waypoint is a list of joint angles.
        title (str): The title for the plot.
    """
    if not trajectory_waypoints:
        print("Cannot plot joint space trajectory: No waypoints provided.")
        return

    num_waypoints = len(trajectory_waypoints)
    # Assuming the first 5 are arm joints, and the 6th is gripper
    num_arm_joints = min(5, len(trajectory_waypoints[0]) -1 if trajectory_waypoints else 0) 

    if num_arm_joints == 0:
        print("Cannot plot joint space trajectory: Waypoints do not seem to contain arm joint data.")
        return

    joint_angles_over_time = np.array([wp[:num_arm_joints] for wp in trajectory_waypoints])

    plt.figure(figsize=(12, 7))
    for i in range(num_arm_joints):
        plt.plot(joint_angles_over_time[:, i], label=f'Joint {i}')
    
    plt.xlabel("Waypoint Index", fontsize=12)
    plt.ylabel("Joint Angle (degrees)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()

def test_trajectory_planning():
    """Test function for trajectory planning and visualization."""
    robot = RobotKinematics(link_lengths=[10, 15, 12, 8, 4])
    print("\nTesting Trajectory Planning...")

    # Define a starting configuration (all joints at a reasonable mid-angle, gripper open)
    # [Base, Shoulder, Elbow, WristPitch, WristRoll, Gripper]
    start_config = [90, 90, 90, 90, 90, 30] # The home position of the robot

    # Define an end target for the end-effector
    end_target_xyz = [10, 10, 20] 
    num_trajectory_waypoints = 30

    print(f"Attempting to plan trajectory from {start_config} to {end_target_xyz}")

    trajectory = robot.plan_trajectory(
        start_config_deg=start_config,
        end_target_xyz=end_target_xyz,
        num_waypoints=num_trajectory_waypoints,
        ik_max_iterations=2000, # Explicitly set higher iterations
        ik_tolerance=0.5 # Use the tolerance that worked well for IK tests
    )

    if trajectory:
        print(f"Trajectory generated with {len(trajectory)} waypoints. Visualizing statically...")
        
        # --- Three-view static plot of EE path and final robot pose ---
        fig_3d = plt.figure(figsize=(18, 6))
        views = [
            {'title': 'Top View', 'init': (90, -90), 'subplot': 131},
            {'title': 'Front View', 'init': (0, -90), 'subplot': 132},
            {'title': 'Side View', 'init': (0, 0), 'subplot': 133}
        ]
        
        final_waypoint_config = trajectory[-1]
        ee_path_xyz_list = []
        for waypoint_cfg in trajectory:
            fk_data = robot.forward_kinematics(waypoint_cfg)
            ee_path_xyz_list.append(fk_data['positions'][6]) # EE is 7th pos (index 6)
        ee_path_xyz_np = np.array(ee_path_xyz_list)

        for view_info in views:
            ax = fig_3d.add_subplot(view_info['subplot'], projection='3d')
            ax.view_init(elev=view_info['init'][0], azim=view_info['init'][1])
            
            # Visualize the robot in its FINAL configuration for this view
            robot.visualize_robot_enhanced(
                final_waypoint_config,
                ax_to_use=ax,
                show_plot=False,
                title_override=None # We'll use subplot title
            )
            
            # Plot the end-effector path
            ax.plot(ee_path_xyz_np[:, 0], ee_path_xyz_np[:, 1], ee_path_xyz_np[:, 2], 'r--', linewidth=2, label='End-Effector Path')
            
            ax.set_title(view_info['title'], fontsize=14, fontweight='bold')
            if view_info['subplot'] == 131: # Add legend to first plot
                ax.legend()

        fig_3d.suptitle("Trajectory Path and Final Pose (3 Views)", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.show()

        # --- Plot illustrating LISP (Joint Space Interpolation) ---
        plot_joint_space_trajectory(trajectory, title="Robot Arm Joint Angles During Trajectory (LISP)")
        
    else:
        print("Failed to generate trajectory.")

def test_inverse_kinematics():
    """Test function for inverse kinematics."""
    robot = RobotKinematics(link_lengths=[10, 15, 12, 8, 4])
    print("Testing Inverse Kinematics...")

    # Define target_points AFTER robot instantiation so we can use robot.link_lengths
    target_points = [
        # Original test points
        [15, 10, 15],    # Likely reachable
        [5, 15, 25],     # Another reachable point
        [30, 0, 10],     # Near the edge of workspace, potentially reachable
        [0, 25, 5],      # Another edge case
        [40, 5, 15],     # Potentially reachable, further out (but caught by simple check now)
        [100, 0, 0],   # Likely unreachable (too far)
        [0, 0, 50],     # Likely unreachable (too high if l1 is short)

        # Additional test points
        [5, 5, 12],      # Close to base, low height
        [-10, 10, 20],   # Negative X, positive Y
        [10, -15, 18],   # Positive X, negative Y
        [-15, -5, 22],   # Negative X, negative Y
        [20, 20, 10],    # Requires significant bending
        # Use robot.link_lengths here as self is not defined in this scope
        [0, 0, robot.link_lengths[0] + sum(robot.link_lengths[1:]) -1], # Max Z reach (approx)
        [38, 0, robot.link_lengths[0] + 1], # Very close to max reach in X, just above base height + a bit
        [0, 38, robot.link_lengths[0] + 1], # Very close to max reach in Y
        [1, 1, robot.link_lengths[0] + sum(robot.link_lengths[1:]) - 2], # Near singularity, almost fully stretched vertically (adjusted)
        [robot.link_lengths[1] + robot.link_lengths[2] - 5, 0, robot.link_lengths[0] + 5] # A point testing mid-reach specifically
    ]

    # Default gripper angle for visualization, not solved by IK
    gripper_angle_for_viz = 70.0 

    for i, target_pos_xyz in enumerate(target_points):
        print(f"\n--- Test Case {i+1}: Attempting to reach target {target_pos_xyz} ---")
        
        # Use default initial guess (mid-range of joint limits)
        # Increase iterations for potentially harder to reach points
        joint_angles_deg_solution = robot.inverse_kinematics(
            target_pos_xyz, 
            initial_guess_angles_deg=None, 
            gripper_angle_deg=gripper_angle_for_viz,
            max_iterations=2000, 
            tolerance=0.5 # Slightly larger tolerance for CCD
        )

        if joint_angles_deg_solution is not None:
            print(f"Solution found: Joint Angles (J0-J4): {[round(a, 2) for a in joint_angles_deg_solution]} degrees")
            
            # Visualize the robot at the found configuration
            # Append the gripper angle for visualization as FK expects 6 angles
            full_config_for_viz = joint_angles_deg_solution + [gripper_angle_for_viz]
            print(f"Visualizing configuration: {[round(a, 2) for a in full_config_for_viz]}")
            robot.visualize_robot_three_views(full_config_for_viz)

            # Optional: Verify by FK if the EE is close to target
            fk_check = robot.forward_kinematics(full_config_for_viz)
            ee_pos_check = fk_check['positions'][6] # EE position is the 7th position (index 6)
            error_check = np.linalg.norm(np.array(target_pos_xyz) - ee_pos_check)
            print(f"FK verification: Reached EE position: {[round(p, 2) for p in ee_pos_check]}, Target: {target_pos_xyz}, Error: {error_check:.3f} cm")
        else:
            print(f"No solution found for target {target_pos_xyz}.")

def main():
    """Test function for robot kinematics."""
    # Create robot with specific link lengths matching a typical robot arm
    # First link is vertical, others are arm segments
    robot = RobotKinematics(link_lengths=[10, 15, 12, 8, 4])
    
    # Test multiple configurations that demonstrate the correct joint movements
    test_configurations = [
        [0, 0, 0, 0, 0, 40],
        [90, 90, 90, 90, 90, 90],
        [0, 45, 45, 45, 45, 0]
    ]
    
    print("Testing multiple robot configurations...")
    for i, config in enumerate(test_configurations):
        print(f"\nConfiguration {i+1}: {config}")
        # Use the three-view visualization
        robot.visualize_robot_three_views(config)
    
    # Visualize workspace
    print("\nAnalyzing and visualizing workspace...")
    robot.visualize_workspace_enhanced(num_samples=200000)  # Reduced for testing

if __name__ == "__main__":
    # main() # Call the original main if needed for other tests
    test_trajectory_planning() # Call the new trajectory planning test function
