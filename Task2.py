import pinocchio as pin
import numpy as np
import time


import numpy as np
import aligator
import pinocchio as pin
import time
import matplotlib.pyplot as plt


from aligator import (
    manifolds,
    dynamics,
    constraints,
)
from utils import ArgsBase


class Args(ArgsBase):
    tcp: str = None
    bounds: bool = True
    num_threads: int = 8


# loadd robot model into pinocchio
try:
    from pinocchio.visualize import MeshcatVisualizer
except ImportError as e:
    raise ImportError("Pinocchio not found, try ``pip install pin``") from e

from robot_descriptions.loaders.pinocchio import load_robot_description


# Load the robot description and initialize the visualizer
robot = load_robot_description("g1_mj_description")
robot.setVisualizer(MeshcatVisualizer())
robot.initViewer(open=True)
robot.loadViewerModel("pinocchio")
robot.display(robot.q0)
# input("\n Press Enter to close MeshCat and terminate... ")

robotComplete=robot


# Parse arguments
args = Args().parse_args()

# Extract model and data from the robot
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
nq = rmodel.nq 
nv = rmodel.nv
nu = nv - 6   # Number of control inputs (excluding floating base)
print("nq:", nq)
print("nv:", nv)


# Define foot frame and joint IDs

FOOT_FRAME_IDS = {
    fname: rmodel.getFrameId(fname) for fname in ["left_ankle_roll_link", "right_ankle_roll_link"]
}
FOOT_JOINT_IDS = {
    fname: rmodel.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
}

# Define controlled joints and their IDs
controlled_joints = rmodel.names[1:].tolist()
controlled_ids = [
    robotComplete.model.getJointId(name_joint) for name_joint in controlled_joints[1:]
]


# Initial configuration
q0 = robot.q0
pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)

# Define the task space
space = manifolds.MultibodyPhaseSpace(rmodel)

# Initial state and control
x0 = np.concatenate((q0, np.zeros(nv)))
u0 = np.zeros(nu)
com0 = pin.centerOfMass(rmodel, rdata, x0[:nq])
dt = 0.01 # Time step



# Define OCP weights
w_x = np.array([
    0, 0, 0, 10000, 10000, 10000,  # Base pos/ori
    1, 1, 1, 1, 1, 1,  # Left leg
    1, 1, 1, 1, 1, 1,  # Right leg
    1000,  # waist
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,  # right arm
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,  # left arm
    100, 100, 100, 100, 100, 100,  # Base pos/ori vel
    10, 10, 10, 10, 10, 10,  # Left leg vel
    10, 10, 10, 10, 10, 10,  # Right leg vel
    1000,  # waist vel
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  # right arm vel
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10  # left arm vel
])
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-3
w_LFRF = 1000000 * np.eye(6)
w_com = 100000 * np.ones(3)
w_com = np.diag(w_com)

# Actuation matrix
act_matrix = np.eye(nv, nu, -6)

# Create dynamics and costs
prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
constraint_models = []
constraint_datas = []
for fname, fid in FOOT_FRAME_IDS.items():
    joint_id = FOOT_JOINT_IDS[fname]
    pl1 = rmodel.frames[fid].placement
    pl2 = rdata.oMf[fid]
    cm = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        rmodel,
        joint_id,
        pl1,
        0,
        pl2,
        pin.LOCAL_WORLD_ALIGNED,
    )
    cm.corrector.Kp[:] = (0, 0, 100, 0, 0, 0)
    cm.corrector.Kd[:] = (50, 50, 50, 50, 50, 50)
    constraint_models.append(cm)
    constraint_datas.append(cm.createData())

# Function to create dynamics based on support phase
def create_dynamics(support):
    dyn_model = None
    if support == "LEFT":
        ode = dynamics.MultibodyConstraintFwdDynamics(
            space, act_matrix, [constraint_models[0]], prox_settings
        )
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    elif support == "RIGHT":
        ode = dynamics.MultibodyConstraintFwdDynamics(
            space, act_matrix, [constraint_models[1]], prox_settings
        )
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    else:
        ode = dynamics.MultibodyConstraintFwdDynamics(
            space, act_matrix, constraint_models, prox_settings
        )
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    return dyn_model

# Define frame IDs and placements
LF_id = rmodel.getFrameId("left_ankle_roll_link")
RF_id = rmodel.getFrameId("right_ankle_roll_link")
root_id = rmodel.getFrameId("pelvis")
LF_placement = rdata.oMf[LF_id]
RF_placement = rdata.oMf[RF_id]

# Define residuals for CoM and frame velocities
frame_com = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com0)
v_ref = pin.Motion()
v_ref.np[:] = 0.0
frame_vel_LF = aligator.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, LF_id, pin.LOCAL
)
frame_vel_RF = aligator.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, RF_id, pin.LOCAL
)

# Function to create a stage in the OCP
def createStage(support, prev_support, LF_target, RF_target,com_target):
    frame_fn_LF = aligator.FramePlacementResidual(
        space.ndx, nu, rmodel, LF_target, LF_id
    )
    frame_fn_RF = aligator.FramePlacementResidual(
        space.ndx, nu, rmodel, RF_target, RF_id
    )
    frame_cs_RF = aligator.FrameTranslationResidual(
        space.ndx, nu, rmodel, RF_target.translation, RF_id
    )[2]
    frame_cs_LF = aligator.FrameTranslationResidual(
        space.ndx, nu, rmodel, LF_target.translation, LF_id
    )[2]


    com_residual = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com_target)

    rcost = aligator.CostStack(space, nu)
    rcost.addCost(aligator.QuadraticStateCost(space, nu, x0, w_x))
    rcost.addCost(aligator.QuadraticControlCost(space, u0, w_u))
    rcost.addCost(aligator.QuadraticResidualCost(space, com_residual, w_com))  # CoM cost

    """ rcost.addCost(aligator.QuadraticResidualCost(space, frame_com, w_com)) """
    if support == "LEFT":
        rcost.addCost(aligator.QuadraticResidualCost(space, frame_fn_RF, w_LFRF))
    elif support == "RIGHT":
        rcost.addCost(aligator.QuadraticResidualCost(space, frame_fn_LF, w_LFRF))

    stm = aligator.StageModel(rcost, create_dynamics(support))
    umax = rmodel.effortLimit[6:]
    umin = -umax
    if args.bounds:
        # print("Control bounds activated")
        # fun: u -> u
        ctrl_fn = aligator.ControlErrorResidual(space.ndx, np.zeros(nu))
        stm.addConstraint(ctrl_fn, constraints.BoxConstraint(umin, umax))

    if support == "DOUBLE" and prev_support == "LEFT":
        stm.addConstraint(frame_vel_RF, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_cs_RF, constraints.EqualityConstraintSet())
    elif support == "DOUBLE" and prev_support == "RIGHT":
        stm.addConstraint(frame_vel_LF, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_cs_LF, constraints.EqualityConstraintSet())

    return stm

# Define terminal cost
term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, 100 * w_x))
""" term_cost.addCost(aligator.QuadraticResidualCost(space, frame_com, 100 * w_com)) """

# Define contact phases and walk parameters
T_ds = 5  # Double support phase duration
T_ss = 60  # Single support phase duration
swing_apex = 0.1  # Maximum height of the swing foot
T_squat = 40  # Squat phase duration
T_stance = 60  # Stance phase duration
squat_depth = 0.1  # Depth of the squat
step_length = 0.2  # Desired forward step length (in meters)


 # Function to generate swing foot trajectory
def ztraj(swing_apex, t_ss, ts, forward_step=0.01):
    z = swing_apex * np.sin(ts / t_ss * np.pi)   
    return z


# Define contact phases
contact_phases = (
    ["DOUBLE"] * T_ds
    + ["LEFT"] * T_ss
    + ["DOUBLE"] * T_ds
    + ["RIGHT"] * T_ss
    + ["DOUBLE"] * T_ds
    + ["LEFT"] * T_ss
    + ["DOUBLE"] * T_ds
    + ["RIGHT"] * T_ss
    + ["DOUBLE"] * T_ds
    + ["LEFT"] * T_ss
    + ["DOUBLE"] * T_ds
    + ["SQUAT"] * T_squat  # Squat phase
    + ["DOUBLE"] * T_ds
    + ["LEFT"] * T_ss
    + ["DOUBLE"] * T_ds
    + ["RIGHT"] * T_ss
    + ["DOUBLE"] * T_ds
    + ["LEFT"] * T_ss
    + ["DOUBLE"] * T_ds
    + ["STANCE"] * T_stance # Stance phase
    
)

# Initialize foot placements and CoM trajectory
LF_placements = []
RF_placements = []
nsteps = len(contact_phases)

com_trajectory = []  # Store CoM trajectory

  
ts = 0
lf_forward_progress = 0.0  # Forward progress for the left foot
rf_forward_progress = 0.0  # Forward progress for the right foot
forward_step = 0.005  # Forward step size for each phase
com_hight = com0[2] - 0.05
lu_step = 0.005

LF_placements.append(LF_placement)
RF_placements.append(RF_placement)
com_trajectory.append(np.array([com0[0], com0[1], com0[2] - 0.05]))  # Initial CoM position

count_rf_ticks = 0

# Generate foot placements and CoM trajectory
for cp in contact_phases:
    ts += 1
    if cp == "DOUBLE":
        ts = 0
        LF_placements.append(LF_placements[-1])
        RF_placements.append(RF_placements[-1])
        com_trajectory.append(np.array([(RF_placements[-1].translation[0]+LF_placements[-1].translation[0])/2.0, com_trajectory[-1][1], com_trajectory[-1][2]]))

    elif cp == "RIGHT":
        LF_goal = LF_placements[-1].copy()

        lf_forward_progress += 2*forward_step
        LF_goal.translation[0] = lf_forward_progress # Accumulated forward translation
        LF_goal.translation[2] = ztraj(swing_apex, T_ss, ts, forward_step) 

        LF_placements.append(LF_goal)
        RF_placements.append(RF_placements[-1])
        com_trajectory.append(np.array([(LF_goal.translation[0]+RF_placements[-1].translation[0])/2.0, com_trajectory[-1][1], com_trajectory[-1][2]]))
            


    elif cp == "LEFT":
        if (count_rf_ticks<=T_ss or ( count_rf_ticks>=2*T_ss )):

            RF_goal = RF_placements[-1].copy()
            rf_forward_progress += forward_step
            RF_goal.translation[0] = rf_forward_progress  # Accumulated forward translation
            RF_goal.translation[2] = ztraj(swing_apex, T_ss, ts, forward_step)         

            LF_placements.append(LF_placements[-1])
            RF_placements.append(RF_goal)
            com_trajectory.append(np.array([(RF_goal.translation[0]+LF_placements[-1].translation[0])/2.0, com_trajectory[-1][1], com_trajectory[-1][2]]))

        else:
            RF_goal = RF_placements[-1].copy()
            rf_forward_progress += 2*forward_step
            RF_goal.translation[0] = rf_forward_progress    # Accumulated forward translation
            RF_goal.translation[2] = ztraj(swing_apex, T_ss, ts, forward_step)            

            LF_placements.append(LF_placements[-1])
            RF_placements.append(RF_goal)
            com_trajectory.append(np.array([(RF_goal.translation[0]+LF_placements[-1].translation[0])/2.0, com_trajectory[-1][1],com_trajectory[-1][2]]))
        count_rf_ticks+=1

    elif cp == "SQUAT":
        LF_placements.append(LF_placements[-1])
        RF_placements.append(RF_placements[-1])

        com_hight-=lu_step
        com_trajectory.append(np.array([(RF_placements[-1].translation[0]+LF_placements[-1].translation[0])/2.0, com_trajectory[-1][1], com_hight]))

    elif cp == "STANCE":
        LF_placements.append(LF_placements[-1])
        RF_placements.append(RF_placements[-1])
        com_hight+=lu_step
        com_trajectory.append(np.array([(RF_placements[-1].translation[0]+LF_placements[-1].translation[0])/2.0, com_trajectory[-1][1], com_hight]))


# Create stages for the OCP
stages = [createStage(contact_phases[0], "DOUBLE", LF_placements[0], RF_placements[0],com_trajectory[0])]
for i in range(1, nsteps):
    stages.append(
        createStage(
            contact_phases[i], contact_phases[i - 1], LF_placements[i], RF_placements[i],com_trajectory[i]
        )
    )


# Define the trajectory optimization problem
problem = aligator.TrajOptProblem(x0, stages, term_cost)

# Solver settings
TOL = 1e-4
mu_init = 1e-8
max_iters = 500
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
solver.max_iters = max_iters
solver.sa_strategy = aligator.SA_FILTER
solver.filter.beta = 1e-5
solver.force_initial_condition = True
solver.reg_min = 1e-6
solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
solver.setNumThreads(args.num_threads)
solver.setup(problem)


# Initial guess for states and controls
us_init = [np.zeros(nu)] * nsteps
xs_init = [x0] * (nsteps + 1)


# Solve the problem
solver.run(problem, xs_init, us_init)
workspace = solver.workspace
results = solver.results
print(results)

def fdisplay():
    qs = [x[:nq] for x in results.xs.tolist()]

    for _ in range(3):
        robot.play(qs, dt)
        time.sleep(0.5)
    # robot.play(qs, dt)
    # time.sleep(0.5)
 
fdisplay()

 
print(len(results.xs.tolist()))
# Extract joint positions, velocities, and torques
joint_positions = [x[:nq] for x in results.xs.tolist()]  # Joint angular positions
joint_velocities = [x[nq:] for x in results.xs.tolist()]  # Joint angular velocities
joint_torques = results.us.tolist()  # Joint torques



# Time vector for plotting
time_vector = np.arange(0, len(joint_positions) * dt, dt)

# # Plot Joint Angular Positions
# plt.figure(figsize=(12, 8))
# for i in range(nq):
#     plt.plot(time_vector, [q[i] for q in joint_positions], label=f'Joint {i+1}')
# plt.title('Joint Angular Positions')
# plt.xlabel('Time (s)')
# plt.ylabel('Position (rad)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot Joint Angular Velocities
# plt.figure(figsize=(12, 8))
# for i in range(nv):
#     plt.plot(time_vector, [v[i] for v in joint_velocities], label=f'Joint {i+1}')
# plt.title('Joint Angular Velocities')
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (rad/s)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot Joint Torques
# plt.figure(figsize=(12, 8))
# for i in range(nu):
#     plt.plot(time_vector[:-1], [u[i] for u in joint_torques], label=f'Joint {i+1}')
# plt.title('Joint Torques')
# plt.xlabel('Time (s)')
# plt.ylabel('Torque (Nm)')
# plt.legend()
# plt.grid(True)
# plt.show()

 

left_leg_joints = range(7, 13)  # Indices for leg joints
right_leg_joints = range(13, 19)  # Indices for leg joints




plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for i in left_leg_joints:
    plt.plot(time_vector, [q[i] for q in joint_positions], label=f'Joint {i+1}')
plt.title('Left Leg Joint Angular Positions')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for i in right_leg_joints:
    plt.plot(time_vector, [q[i] for q in joint_positions], label=f'Joint {i+1}')
plt.title('Right Leg Joint Angular Positions')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for i in left_leg_joints:
    plt.plot(time_vector, [v[i-1] for v in joint_velocities], label=f'Joint {i+1}')
plt.title('Left Leg Joint Angular Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for i in right_leg_joints:
    plt.plot(time_vector, [v[i-1] for v in joint_velocities], label=f'Joint {i+1}')
plt.title('Right Leg Joint Angular Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




left_leg_torques= range(0, 6)  # Indices for leg joints
right_leg_torques = range(6, 12)  # Indices for leg joints



plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for i in left_leg_torques:
    plt.plot(time_vector, [u[i] for u in joint_velocities], label=f'Joint {i+1}')
plt.title('Left Leg Joint Torques')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for i in right_leg_torques:
    plt.plot(time_vector, [u[i] for u in joint_velocities], label=f'Joint {i+1}')
plt.title('Right Leg Joint Torques')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()