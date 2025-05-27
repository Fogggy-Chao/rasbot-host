import random, time, json
import threading
import queue

# Attempt to import the local IK solver
try:
    from local_kinematics import solve_inverse_kinematics
    LOCAL_IK_AVAILABLE = True
except ImportError:
    LOCAL_IK_AVAILABLE = False
    print("[RPIExecutor] Warning: local_kinematics.py not found or RobotKinematics could not be imported. Local IK calls will fail.")
    def solve_inverse_kinematics(*args, **kwargs): # Placeholder if import fails
        return {"status": "error", "message": "Local IK solver not available due to import error."}

class ExecutorBase:
    """Route LLM tool calls to the real robot or to mocks."""
    def __call__(self, name: str, args: dict) -> dict:
        raise NotImplementedError

class MockExecutor(ExecutorBase):
    """Fast deterministic mocks for local dev / unit tests."""
    def __call__(self, name: str, args: dict) -> dict:
        print(f"[MOCK] {name}({args})")
        time.sleep(0.2)               # pretend the robot did something
        if name == "detect_object":
            # fake a cup somewhere in front
            object_name = args.get("object_name", "unknown")
            print(f"[MOCK] Detecting {object_name}...")
            # Return coordinates slightly randomized
            return {
                "x": round(random.uniform(30, 50), 2),
                "y": round(random.uniform(-10, 10), 2),
                "z": round(random.uniform(100, 200), 2),
                "confidence": round(random.uniform(0.8, 0.95), 2)
            }
        if name == "inverse_kinematics":
            # fake some joint angles
            print(f"[MOCK] Calculating IK for position {args.get('object_position')}...")
            return {"joint_angles": [round(random.uniform(-1.5, 1.5), 2) for _ in range(6)]}
        if name == "stop":
             print(f"[MOCK] Stopping robot...")
             return {"status": "ok"}
        if name in {"move_base", "move_arm", "rotate_base", "grasp"}:
            print(f"[MOCK] Executing {name}...")
            return {"status": "ok"}
        # Default for unknown tools
        print(f"[MOCK] Warning: Unknown tool '{name}' called.")
        return {"status": "unknown_tool", "tool_name": name}

class RPIExecutor(ExecutorBase):
    """Send JSON over your existing websocket to the Pi."""
    def __init__(self, ws_rpi):
        self.ws = ws_rpi
        self.reply_queue = queue.Queue(maxsize=1)
        self.reply_event = threading.Event()
        
        # Assign the on_message callback directly to the WebSocketApp instance
        if self.ws:
            self.ws.on_message = self._on_message_callback

    def _on_message_callback(self, wsapp, message_str):
        """Callback for when a message is received from RPI."""
        try:
            # Clear previous message if any, to avoid processing stale data
            # This could happen if a previous call timed out before a message arrived
            try:
                self.reply_queue.get_nowait()
            except queue.Empty:
                pass # Queue was already empty

            self.reply_queue.put(json.loads(message_str))
            self.reply_event.set()
        except json.JSONDecodeError:
            print(f"[RPIExecutor] Error decoding JSON from RPI: {message_str}")
            # Optionally, put an error object in queue or handle differently
        except Exception as e:
            print(f"[RPIExecutor] Unexpected error in on_message_callback: {e}")


    def __call__(self, name: str, args: dict) -> dict:
        if not self.ws or not self.ws.sock or not self.ws.sock.connected:
            print("[RPIExecutor] WebSocket not connected.")
            return {"status": "error", "message": "WebSocket not connected to RPI"}

        # Check if this is an inverse_kinematics call and if local IK is available
        if name == "inverse_kinematics" and LOCAL_IK_AVAILABLE:
            target_position_mm = args.get("object_position") # Matches system_prompt.txt example, in mm
            if target_position_mm is None:
                return {"status": "error", "message": "Missing 'object_position' for inverse_kinematics"}
            
            # Convert target_position from mm to cm
            if isinstance(target_position_mm, list) and len(target_position_mm) == 3:
                target_position_cm = [coord / 10.0 for coord in target_position_mm]
            else:
                return {"status": "error", "message": "Invalid 'object_position' format. Expected list of 3 numbers."}

            # Potentially get gripper_angle from args if specified by LLM, otherwise use default in solve_inverse_kinematics
            gripper_angle = args.get("gripper_angle") # This arg name is an assumption
            initial_guess = args.get("initial_guess_angles") # This arg name is an assumption

            print(f"[RPIExecutor] Calling local IK. Original mm: {target_position_mm}, Converted cm: {target_position_cm}")

            if gripper_angle is not None:
                return solve_inverse_kinematics(target_position_cm, initial_guess_angles_deg=initial_guess, gripper_angle_deg=gripper_angle)
            else:
                return solve_inverse_kinematics(target_position_cm, initial_guess_angles_deg=initial_guess)

        # Original behavior for other tools or if local IK is not available
        payload = json.dumps({"name": name, "arguments": args})
        
        self.reply_event.clear() # Clear the event before sending
        # Ensure queue is clear of stale messages from previous calls
        try:
            self.reply_queue.get_nowait()
        except queue.Empty:
            pass

        self.ws.send(payload)
        
        # Wait for the on_message callback to signal a reply
        if self.reply_event.wait(timeout=10.0):  # Timeout after 10 seconds
            try:
                reply = self.reply_queue.get_nowait() # Use get_nowait as event confirmed message
                return reply
            except queue.Empty:
                print("[RPIExecutor] Event was set but queue is empty. This should not happen.")
                return {"status": "error", "message": "Internal error: Event set but no message in queue"}
        else:
            print(f"[RPIExecutor] Timeout waiting for reply from RPI for tool: {name}")
            return {"status": "error", "message": f"Timeout waiting for RPI reply for tool: {name}"}