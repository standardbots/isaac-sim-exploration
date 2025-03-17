import traceback
import sys
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger("TeleoperateScript")
logger.info("Script started - initializing simulation")

try:
    # Verified correct import for SimulationApp
    from isaacsim.simulation_app import SimulationApp
    
    # Initialize simulation app first
    simulation_app = SimulationApp({"headless": True})
    
    # Add WebRTC server initialization
    from isaacsim.kit.webrtc.server import WebRTCServer
    webrtc_server = WebRTCServer(port=8211)  # You can choose a different port if needed
    webrtc_server.start()
    
    # Updated imports based on documentation
    import omni.usd
    import carb.input
    
    # These are now in the isaacsim namespace
    from isaacsim.core.world import World  # Updated from omni.isaac.core
    from isaacsim.core.articulations.articulation import Articulation  # Updated path
    from isaacsim.kit.viewport.capture import ViewportCapture  # Updated path
    from isaacsim.kit.viewport.offscreen_viewport import OffscreenViewport
    from isaacsim.kit.viewport.capture import OffscreenCapture
    
    print("All imports successful!")
except Exception as e:
    print(f"Error importing modules: {e}")
    print(traceback.format_exc())
    exit(1)

# Create world
world = World()

# Import USD file - replace with your USD file path
usd_path = "/path/to/your/robot.usd"  # Change this to your USD file path
robot_prim_path = "/robot"  # This will be determined by your USD structure

# Add USD to the stage
omni.usd.get_context().open_stage(usd_path)

# Wait for physics to initialize and then reset
world.reset()

# Get robot articulation
# Note: You may need to adjust the prim_path based on your USD file structure
robot = world.scene.add(Articulation(prim_path=robot_prim_path))

# Set joint drive modes: velocity for arm, position for gripper
joint_names = robot.get_joint_names()
arm_joint_names = joint_names[:6]  # First 6 joints for the arm (adjust if different)
gripper_joint_name = joint_names[6]  # Gripper joint (adjust index if different)
for joint_name in arm_joint_names:
    joint_path = f"{robot_prim_path}/{joint_name}"
    prim = usd_utils.get_stage().GetPrimAtPath(joint_path)
    prim.GetAttribute("drive:angular:physics:mode").Set("velocity")  # Revolute joints for arm
gripper_joint_path = f"{robot_prim_path}/{gripper_joint_name}"
gripper_prim = usd_utils.get_stage().GetPrimAtPath(gripper_joint_path)
gripper_prim.GetAttribute("drive:linear:physics:mode").Set("position")  # Prismatic joint for gripper

# Find end effector index
ee_body_name = "ee_link"  # Adjust to your end effector link name from USD
body_names = robot.get_body_names()
ee_index = body_names.index(ee_body_name)

# Set up input interface for space mouse and keyboard
input_interface = carb.input.acquire_input_interface()
space_mouse = input_interface.get_input_device("space_mouse")

# Keyboard event handler for gripper control and quitting
quit_flag = False
is_gripper_closed = False
open_pos = 0.0    # Gripper open position (adjust based on URDF joint range)
close_pos = 0.05  # Gripper closed position (adjust based on URDF joint range)

def on_keyboard_event(event):
    global quit_flag, is_gripper_closed
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.G:
            is_gripper_closed = not is_gripper_closed
            print(f"Gripper {'closed' if is_gripper_closed else 'open'}")
        elif event.input == carb.input.KeyboardInput.Q:
            quit_flag = True

input_interface.subscribe_to_keyboard_events(on_keyboard_event)

# Space mouse scaling factors
scale_linear = 0.1   # Linear velocity scale (m/s)
scale_angular = 0.5  # Angular velocity scale (rad/s)

# Create offscreen viewport
#viewport = OffscreenViewport(width=1920, height=1080)
#capture = OffscreenCapture(viewport)
#capture.start_recording("output.mp4", fps=60)

# Main teleoperation loop
while not quit_flag:
    # Get space mouse input
    state = input_interface.get_input_device_state(space_mouse)
    if state is not None and hasattr(state, 'axes'):
        dx = state.axes[0] * scale_linear  # Translation X
        dy = state.axes[1] * scale_linear  # Translation Y
        dz = state.axes[2] * scale_linear  # Translation Z
        drx = state.axes[3] * scale_angular  # Rotation X
        dry = state.axes[4] * scale_angular  # Rotation Y
        drz = state.axes[5] * scale_angular  # Rotation Z
        desired_linear_vel = np.array([dx, dy, dz])
        desired_angular_vel = np.array([drx, dry, drz])
    else:
        desired_linear_vel = np.zeros(3)
        desired_angular_vel = np.zeros(3)

    # Compute differential IK using Jacobian
    jacobian = robot.get_jacobians()[ee_index]  # Shape: (6, num_dofs)
    pinv_jacobian = np.linalg.pinv(jacobian)
    desired_ee_vel = np.concatenate((desired_linear_vel, desired_angular_vel))
    joint_vel = pinv_jacobian @ desired_ee_vel

    # Apply velocities to arm joints (first 6), set gripper velocity to 0
    full_joint_vel = np.zeros(robot.num_dof)
    full_joint_vel[:6] = joint_vel[:6]
    robot.set_joint_velocities(full_joint_vel)

    # Control gripper position
    desired_gripper_pos = close_pos if is_gripper_closed else open_pos
    robot.set_joint_positions([desired_gripper_pos], joint_indices=[6])  # Index 6 for gripper

    # Advance simulation
    world.step()

# Stop video recording
capture.stop_recording()

# Cleanup
webrtc_server.stop()
simulation_app.close()
logger.info("Script completed successfully")
