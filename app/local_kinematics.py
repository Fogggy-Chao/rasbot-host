import sys
import os
import numpy as np

# Add the simulation directory to the Python path to import RobotKinematics
# Assuming app and simulation are sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
simulation_dir = os.path.join(parent_dir, 'simulation')
if simulation_dir not in sys.path:
    sys.path.append(simulation_dir)

try:
    from kinematics import RobotKinematics
except ImportError as e:
    print(f"Error importing RobotKinematics: {e}")
    print(f"Current sys.path: {sys.path}")
    # Fallback for environments where sys.path manipulation might not be ideal
    # This assumes kinematics.py is directly in the simulation folder relative to project root
    try:
        from simulation.kinematics import RobotKinematics
    except ImportError:
        raise ImportError("Could not import RobotKinematics. Ensure 'simulation' directory is accessible.")

# Initialize the robot kinematics model
# You can specify link_lengths if different from default: [10, 15, 12, 8, 4]
robot_kinematics_model = RobotKinematics()

DEFAULT_GRIPPER_ANGLE_DEG = 70.0 # As used in kinematics.py IK

def solve_inverse_kinematics(target_position_xyz: list[float], 
                               initial_guess_angles_deg: list[float] | None = None, 
                               gripper_angle_deg: float = DEFAULT_GRIPPER_ANGLE_DEG) -> dict:
    """
    Solves inverse kinematics using the local RobotKinematics model.

    Args:
        target_position_xyz: Target [x, y, z] coordinates for the end-effector.
        initial_guess_angles_deg: Optional initial guess for arm joint angles [J0-J4] in degrees.
        gripper_angle_deg: Angle for the gripper (J5) in degrees.

    Returns:
        A dictionary with either 'joint_angles' (list of 6 angles) on success,
        or 'status' and 'message' on failure.
    """
    if not isinstance(target_position_xyz, (list, np.ndarray)) or len(target_position_xyz) != 3:
        return {"status": "error", "message": "Invalid target_position_xyz: Must be a list of 3 floats."}

    print(f"[LocalKinematics] Solving IK for target: {target_position_xyz}, gripper: {gripper_angle_deg}°")

    # The IK solver in kinematics.py expects 5 initial guess angles (J0-J4)
    initial_arm_guess = None
    if initial_guess_angles_deg and len(initial_guess_angles_deg) >= 5:
        initial_arm_guess = initial_guess_angles_deg[:5]
    
    arm_joint_angles_solution_deg = robot_kinematics_model.inverse_kinematics(
        target_position_xyz=target_position_xyz,
        initial_guess_angles_deg=initial_arm_guess,
        gripper_angle_deg=gripper_angle_deg, # Passed for internal consistency if used by IK solver
        max_iterations=5000, # Default from kinematics.py tests
        tolerance=10        # Changed from 0.5 to 6.0 cm
    )

    if arm_joint_angles_solution_deg is not None:
        # IK solver returns 5 angles for J0-J4. Append the gripper angle for J5.
        full_joint_angles = arm_joint_angles_solution_deg + [gripper_angle_deg]
        print(f"[LocalKinematics] IK Solution found: {full_joint_angles}")
        return {"joint_angles": full_joint_angles}
    else:
        print(f"[LocalKinematics] IK Solution not found for target: {target_position_xyz}")
        return {"status": "error", "message": f"Inverse kinematics solution not found for target {target_position_xyz}"}

if __name__ == '__main__':
    # Test the local IK solver
    test_target = [15, 10, 15] # A reachable point from kinematics.py tests
    print(f"Testing local IK for target: {test_target}")
    result = solve_inverse_kinematics(test_target)
    print(f"Result: {result}")

    test_target_unreachable = [100, 0, 0] # Likely unreachable
    print(f"\nTesting local IK for (likely unreachable) target: {test_target_unreachable}")
    result_unreachable = solve_inverse_kinematics(test_target_unreachable)
    print(f"Result: {result_unreachable}")

    # Test with initial guess (assuming 6 angles are provided, we take first 5)
    test_target_2 = [5, 15, 25]
    initial_guess_6_angles = [90, 90, 90, 90, 90, 70]
    print(f"\nTesting local IK for target: {test_target_2} with initial guess (first 5 of {initial_guess_6_angles})")
    result_with_guess = solve_inverse_kinematics(test_target_2, initial_guess_angles_deg=initial_guess_6_angles)
    print(f"Result: {result_with_guess}") 