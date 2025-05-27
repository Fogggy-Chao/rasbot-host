# Tools for the model

TOOLS = [
  {
    "type": "function",
    "name": "detect_object",
    "description": "Run YOLO11n + stereo triangulation to locate a target.",
    "parameters": {
      "type": "object",
      "properties": {
        "object_name": {"type": "string", "description": "The name of the object to detect."}
      },
      "required": ["object_name"]
    }
  },

  {
    "type": "function",
    "name": "inverse_kinematics",
    "description": "Calculate the joint angles to move to a given position.",
    "parameters": {
      "type": "object",
      "properties": {
        "object_position": {
            "type": "array", 
            "description": "The object position (e.g., [x, y, z]) returned by the detect_object tool.",
            "items": {"type": "number"}
         }
      },
      "required": ["object_position"]
    }
  },

  {
    "type": "function",
    "name": "move_arm",
    "description": "Move the arm to a given joint angles.",
    "parameters": {
      "type": "object",
      "properties": {
        "joint_angles": {
            "type": "array", 
            "description": "The joint angles to move to.",
            "items": {"type": "number"}
        }
      },
      "required": ["joint_angles"]
    }
  },

  {
    "type": "function",
    "name": "move_base",
    "description": "Drive the four wheel chassis a given coordinates.",
    "parameters": {
      "type": "object",
      "properties": {
        "coordinates": {
            "type": "array", 
            "description": "The coordinates to move to.",
            "items": {"type": "number"}
        }
      },
      "required": ["coordinates"]
    }
  },

  {
    "type": "function",
    "name": "stop",
    "description": "Stop the robot.",
    "parameters": {}
  },

  {
    "type": "function",
    "name": "grasp",
    "description": "Grasp an object.",
    "parameters": {
      "type": "object",
      "properties": {
          "joint_angle": {"type": "number", "description": "The joint angle of the end effector. It should be within 40 to 100 degrees."}
      },
      "required": ["joint_angle"]
    }
  }, 

  {
    "type": "function",
    "name": "rotate_base",
    "description": "Rotate the base to a given angle.",
    "parameters": {
      "type": "object",
      "properties": {
        "base_angle": {"type": "number", "description": "The angle to rotate the base to."}
      },
      "required": ["base_angle"]
    }
  },

  {
    "type": "function",
    "name": "move_wheels",
    "description": "Controls the robot's chassis wheels for movement. This command initiates the action, and the robot will continue performing it until a new move_wheels command (e.g., with action=\"stop\") or a stop command is issued. For moving specific distances or turning specific angles, you may need to issue this command, then stop after an estimated duration.",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "description": "The movement action. Supported actions:\n\"forward\": Move forward.\n\"backward\": Move backward.\n\"left_turn\": Spin counter-clockwise (turn left on the spot).\n\"right_turn\": Spin clockwise (turn right on the spot).\n\"stop\": Stop wheel movement (can also use the dedicated stop tool).",
          "enum": ["forward", "backward", "left_turn", "right_turn", "stop"]
        },
        "speed": {
          "type": "number",
          "description": "The speed of movement, from 0.0 (stop) to 1.0 (maximum speed). This parameter is usually ignored if action is \"stop\".",
          "minimum": 0.0,
          "maximum": 1.0
        }
      },
      "required": ["action"]
    }
  },
]
