# Experiment: End-to-End Mobile Robot Task Execution

## 1. Objective

To demonstrate the end-to-end capability of the mobile-robot planner in executing a series of foundational, multi-step, natural-language commands using a perceive–plan–act loop with mock data. This showcase will highlight the interaction between real-time transcription, LLM-based planning and tool invocation, the executor layer, and feedback integration, including handling of simulated tool failures.

## 2. System Overview (Conceptual Flowchart)

```mermaid
graph TD
    A[User Speaks Command] --> B{Whisper Transcription};
    B --> C{GPT-4o Planner};
    C -- Tool Call (JSON) --> D{Executor Layer};
    D -- Mock Function Call --> E[Mocked Perception/Control];
    E -- Result --> D;
    D -- Tool Result (JSON) --> C;
    C -- Need More Steps? --> C;
    C -- Task Complete/Error --> F[LLM Text Output (DONE:/ERROR:)];
    F --> G[User Notified];
```

**Figure 1:** Conceptual flowchart of the perceive–plan–act loop. The LLM (GPT-4o Planner) remains in a loop, calling tools via the Executor Layer and receiving results, until the task is complete or an error occurs.

## 3. Experiment Scenarios & Traces

We will trace the execution flow for three distinct scenarios, showing the data passed at each step and simulated latencies.

**Key:**
*   `(W)`: Whisper Transcription
*   `(P)`: GPT-4o Planner
*   `(T)`: Tool Execution (MockExecutor)
*   `L_ transcription`: Latency for transcription
*   `L_planning`: Latency for LLM planning and tool call generation
*   `L_execution`: Latency for tool execution (mocked)
*   `L_feedback`: Latency for sending tool result back to LLM

### 3.1 Experiment 1: Basic Wheel Movement

**User Command:** "Robot, drive forward half a meter."

**Assumed Initial State:** Robot is stationary.

| Step | Action                     | Initiator/Processor | Input / Tool Call                                 | Output / Tool Result          | Simulated Latency (ms) | Cumulative Latency (ms) | Notes                                      |
|------|----------------------------|---------------------|---------------------------------------------------|-------------------------------|------------------------|-------------------------|--------------------------------------------|
| 1.1  | User Speaks                | User                | "Robot, drive forward half a meter."            | Audio Stream                  | -                      | 0                       |                                            |
| 1.2  | Transcribe Audio           | Whisper `(W)`       | Audio Stream                                      | Text: "Robot, drive forward..." | 300 (L_transcription)  | 300                     |                                            |
| 1.3  | Plan & Call Tool           | GPT-4o `(P)`        | Text Prompt                                       | Tool Call: `move_base({"coordinates": [500, 0]})` | 400 (L_planning)       | 700                     | Assumes forward is +X, units are mm.       |
| 1.4  | Execute `move_base`        | Executor `(T)`      | `move_base({"coordinates": [500, 0]})`          | Result: `{"status": "ok"}`     | 250 (L_execution)      | 950                     | Mock base movement.                        |
| 1.5  | Feedback to LLM            | System              | `{"status": "ok"}`                                | -                             | 50 (L_feedback)        | 1000                    |                                            |
| 1.6  | Plan & Finalize            | GPT-4o `(P)`        | Tool Result (Moved)                               | Text Output: "DONE: Moved forward half a meter." | 200 (L_planning)       | 1200                    | Task complete.                             |

**Expected Outcome (Experiment 1):**
*   LLM correctly calls `move_base` with appropriate parameters.
*   LLM outputs a "DONE:" message upon successful completion.
*   Simulated total latency: ~1.2 seconds.

### 3.2 Experiment 2: Basic Arm Movement (Wave)

**User Command:** "Robot, wave your arm."

**Assumed Initial State:** Robot arm is at a neutral position.

| Step | Action                     | Initiator/Processor | Input / Tool Call                                       | Output / Tool Result          | Simulated Latency (ms) | Cumulative Latency (ms) | Notes                                           |
|------|----------------------------|---------------------|---------------------------------------------------------|-------------------------------|------------------------|-------------------------|-------------------------------------------------|
| 2.1  | User Speaks                | User                | "Robot, wave your arm."                                 | Audio Stream                  | -                      | 0                       |                                                 |
| 2.2  | Transcribe Audio           | Whisper `(W)`       | Audio Stream                                            | Text: "Robot, wave your arm."   | 250 (L_transcription)  | 250                     |                                                 |
| 2.3  | Plan & Call Tool           | GPT-4o `(P)`        | Text Prompt                                             | Tool Call: `move_arm({"joint_angles": [0,0,45,0,0,0]})` | 450 (L_planning)       | 700                     | Example: Moves 3rd joint for a simple wave part 1. |
| 2.4  | Execute `move_arm`         | Executor `(T)`      | `move_arm(...)`                                         | Result: `{"status": "ok"}`     | 200 (L_execution)      | 900                     | Mock arm movement.                              |
| 2.5  | Feedback to LLM            | System              | `{"status": "ok"}`                                      | -                             | 50 (L_feedback)        | 950                     |                                                 |
| 2.6  | Plan & Call Tool           | GPT-4o `(P)`        | Tool Result (Moved)                                     | Tool Call: `move_arm({"joint_angles": [0,0,-45,0,0,0]})`| 400 (L_planning)       | 1350                    | Moves 3rd joint back for wave part 2.           |
| 2.7  | Execute `move_arm`         | Executor `(T)`      | `move_arm(...)`                                         | Result: `{"status": "ok"}`     | 200 (L_execution)      | 1550                    | Mock arm movement.                              |
| 2.8  | Feedback to LLM            | System              | `{"status": "ok"}`                                      | -                             | 50 (L_feedback)        | 1600                    |                                                 |
| 2.9  | Plan & Finalize            | GPT-4o `(P)`        | Tool Result (Moved)                                     | Text Output: "DONE: Waved arm." | 200 (L_planning)       | 1800                    | Task complete.                                  |

**Expected Outcome (Experiment 2):**
*   LLM correctly calls `move_arm` twice (or more for a complex wave) to simulate a wave.
*   LLM outputs a "DONE:" message.
*   Simulated total latency: ~1.8 seconds.

### 3.3 Experiment 3: Find Object & Attempt Failed Grasp

**User Command:** "Robot, find the blue cup and try to grab it."

**Assumed Initial State:** A blue cup is on a table within reach.

| Step | Action                       | Initiator/Processor | Input / Tool Call                                           | Output / Tool Result                                               | Simulated Latency (ms) | Cumulative Latency (ms) | Notes                                                        |
|------|------------------------------|---------------------|-------------------------------------------------------------|--------------------------------------------------------------------|------------------------|-------------------------|--------------------------------------------------------------|
| 3.1  | User Speaks                  | User                | "Robot, find the blue cup and try to grab it."            | Audio Stream                                                       | -                      | 0                       |                                                              |
| 3.2  | Transcribe Audio             | Whisper `(W)`       | Audio Stream                                                | Text: "Robot, find the blue cup..."                              | 300 (L_transcription)  | 300                     |                                                              |
| 3.3  | Plan & Call Tool             | GPT-4o `(P)`        | Text Prompt                                                 | Tool Call: `detect_object({"object_name": "blue cup"})`           | 500 (L_planning)       | 800                     | LLM finds the cup.                                           |
| 3.4  | Execute `detect_object`      | Executor `(T)`      | `detect_object(...)`                                        | Result: `{"x": 0.6, "y": -0.1, "z": 0.7, "confidence": 0.88}` | 200 (L_execution)      | 1000                    | Mock cup found.                                              |
| 3.5  | Feedback to LLM              | System              | `{"x": 0.6, ...}`                                           | -                                                                  | 50 (L_feedback)        | 1050                    |                                                              |
| 3.6  | Plan & Call Tool             | GPT-4o `(P)`        | Tool Result (Cup Pose)                                      | Tool Call: `inverse_kinematics({"object_position": [0.6, -0.1, 0.7]})` | 450 (L_planning)       | 1500                    | LLM calculates arm movement.                                 |
| 3.7  | Execute `inverse_kinematics` | Executor `(T)`      | `inverse_kinematics(...)`                                   | Result: `{"joint_angles": [0.2, 0.1, -0.2, 0.5, 0.3, -0.3]}`     | 150 (L_execution)      | 1650                    | Mock IK solution.                                            |
| 3.8  | Feedback to LLM              | System              | `{"joint_angles": ...}`                                      | -                                                                  | 50 (L_feedback)        | 1700                    |                                                              |
| 3.9  | Plan & Call Tool             | GPT-4o `(P)`        | Tool Result (Joint Angles)                                  | Tool Call: `move_arm({"joint_angles": [0.2, ..., -0.3]})`       | 400 (L_planning)       | 2100                    | LLM commands arm to move.                                    |
| 3.10 | Execute `move_arm`           | Executor `(T)`      | `move_arm(...)`                                             | Result: `{"status": "ok"}`                                       | 300 (L_execution)      | 2400                    | Mock arm movement successful.                                |
| 3.11 | Feedback to LLM              | System              | `{"status": "ok"}`                                          | -                                                                  | 50 (L_feedback)        | 2450                    |                                                              |
| 3.12 | Plan & Call Tool             | GPT-4o `(P)`        | Tool Result (Arm Moved)                                     | Tool Call: `grasp({"joint_angle": 60})`                          | 350 (L_planning)       | 2800                    | LLM commands grasp.                                          |
| 3.13 | Execute `grasp`              | Executor `(T)`      | `grasp({"joint_angle": 60})`                                | Result: `{"status": "error", "reason": "Grasp failed, object not secured"}` | 250 (L_execution)      | 3050                    | **Mock grasp fails as intended.**                            |
| 3.14 | Feedback to LLM              | System              | `{"status": "error", "reason": ...}`                         | -                                                                  | 50 (L_feedback)        | 3100                    |                                                              |
| 3.15 | Plan & Finalize              | GPT-4o `(P)`        | Tool Result (Grasp Failed)                                  | Text Output: "ERROR: Attempted to grasp the blue cup, but the grasp was not successful." | 300 (L_planning)       | 3400                    | LLM reports the error.                                       |

**Expected Outcome (Experiment 3):**
*   LLM correctly chains `detect_object`, `inverse_kinematics`, `move_arm`, and `grasp`.
*   The `grasp` tool returns a simulated error.
*   LLM outputs an "ERROR:" message reflecting the grasp failure.
*   Simulated total latency: ~3.4 seconds.

## 4. Visualization of Latency (Conceptual)

*(This would be a bar chart or timeline graph if we could render images)*

**Figure 2:** Conceptual Latency Breakdown per Step (for each experiment).
This figure would show a stacked bar for each step in the tables above, with segments representing `L_transcription`, `L_planning`, `L_execution`, and `L_feedback` where applicable. This helps visualize where time is spent in the loop for each scenario.

## 5. Discussion

These simulated traces for simpler tasks demonstrate the intended foundational functionalities:
*   **Natural Language Understanding:** Whisper transcribes various user commands.
*   **Intent Decomposition & Planning:** GPT-4o breaks down commands into sequences of callable tools, including multi-step sequences for tasks like "wave arm."
*   **Tool Execution & Feedback:** The system correctly calls mock tools and processes their success or failure results.
*   **Error Handling:** The system demonstrates the ability to receive an error from a tool and have the LLM report this error to the user (as in Experiment 3).
*   **State Management:** The LLM implicitly manages state by using the results of previous tool calls to inform subsequent ones.
*   **Modularity:** The separation between the LLM (high-level reasoning) and the executor (low-level control) remains clear.

These experiments, even with mock data, provide a strong indication of the system's potential for robust, multi-step task execution and basic error reporting. Further testing with the `RPIExecutor` and real hardware would validate these findings in a physical environment.

--- 