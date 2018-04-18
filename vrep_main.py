import time 
import sys
import os
import numpy as np
import matplotlib as mlp
import subprocess
import psutil

try:
    import vrep
except:
    print ('--------------------------------------')
    print ('Error: Importing vrep.py failed')
    print ('make sure vrep.py, vrepConst.py,')
    print ('vrepConst.py and remoteApi.so are')
    print ('in your working directory')
    print ('--------------------------------------')

propeller_vel_signal = 'objectIdAndForce'
client_id = 0
process = None 

def start_vrep_subprocess_or_die(vrep_path="/home/thomas/V-REP/"):
    global process
    vrep_script = os.path.join(vrep_path, "vrep.sh")
    process = subprocess.Popen(["bash", vrep_script])
    print("Initializing V-REP!")
    time.sleep(7.0)

def kill_vrep_subprocess():
    p = psutil.Process(process.pid)
    for proc in p.children(recursive=True):
        proc.kill()
    p.kill()

def init_vrep_connection_or_die():
    global client_id
    # --------------------------------------------------------
    vrep.simxFinish(-1) # close any opened conenctions 
    client_id = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if client_id == -1:
        print 'Failed to connect to remote API server'
        sys.exit('could not connect')
    
    print ('Connection to Remote Api Server: Established')
    return client_id

def load_quad_scene_or_die(): 
    # SUPRISE ... it never dies :)
    try:
        _, _ = vrep.simxLoadScene(client_id, 'vrep_scenes/modified_quad.ttt', 0xFF, vrep.simx_opmode_blocking)
    except:
        print "simxLoadScene throws an exception but works. Why? we don't know..."

def get_quadcopter_handle_or_die():
    global quad_handle
    # -------------------------------------------------------
    res, qh = vrep.simxGetObjectHandle(client_id, "Quadricopter_base", vrep.simx_opmode_blocking)
    if res != vrep.simx_return_ok:
        print 'Could not GetObjectHandle: Quadricopter_base'
        sys.exit('Failed to grab Quadricopter_base handle')
    quad_handle = qh
    print ('grabbed Quadricopter_base handle ', quad_handle)

    # ------------------------------------------------------
    #res, cam_handle = vrep.simxGetObjectHandle(client_id, "CAMERA_ID", vrep.simx_opmode_blocking)
    #if res != simx_return_ok:
    #    print "Could not access camera object."
    #    sys.exit('Failed to access V-REP Vision Sensor Object')
    #print ('recieved camera object handle', cam_handle)
    # ------------------------------------------------------

def init_sim_or_die():
    start_vrep_subprocess_or_die()
    init_vrep_connection_or_die()
    load_quad_scene_or_die()
    get_quadcopter_handle_or_die()


# takes a float array of rotor velocities
def set_propeller_velocities(prop_vel):
    global client_id
    data = vrep.simxPackFloats(prop_vel)
    vrep.simxSetStringSignal(client_id, propeller_vel_signal, data, vrep.simx_opmode_oneshot)


def setpoint(thrust, roll, pitch, yaw):
    global client_id
    data = vrep.simxPackFloats([thrust, roll, pitch, yaw])
    vrep.simxSetStringSignal(client_id, propeller_vel_signal, data, vrep.simx_opmode_oneshot)

def setpoint_(trpy):
    setpoint(trpy[0], trpy[1], trpy[2], trpy[3])

def reset_sim():
    global goal
    global timestep 
    
    print "reset sim"
    ret = vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
    goal = np.array([2.0, 2.0, 1.0])
    timestep = 0


def stop_sim():
    _ = vrep.simxStopSimulation(client_id, vrep.simx_opmode_blocking)
    time.sleep(0.1)

states = ["not_started", "simulating", "done"]
floor_collision_threshold = 0.09

pos = np.array([0.0, 0.0, 0.0])
eul = np.array([0.0, 0.0, 0.0])
last_pos = np.array([0.0, 0.0, 0.0])
last_eul = np.array([0.0, 0.0, 0.0])
goal =  np.array([0.0, 0.0, 0.0])
timestep = 0
max_timesteps = 100

######################################################3333
# step (action) returns next_state, reward, done
# action is a numpy vector in R^4 (thrust, roll, pitch, yaw)
# next_state is a numpy vector in R^15 (pose, velocity, orientation, angular_vel, goal_relative_to_pose)
# reward g(state, action) is a scalar reward 
# done boolean is the simulation done 
def step(action):
    global pos
    global eul 
    global last_pos
    global last_eul
    global goal 
    global max_timesteps
    global timestep

    timestep += 1
    setpoint(action[0], action[1], action[2], action[3])
    last_pos = pos 
    last_eul = eul 
    ret_p, pos = vrep.simxGetObjectPosition(client_id, quad_handle, -1, vrep.simx_opmode_blocking)
    ret_e, eul = vrep.simxGetObjectOrientation(client_id, quad_handle, -1, vrep.simx_opmode_blocking)
    
    while ret_p != vrep.simx_return_ok or ret_e != vrep.simx_return_ok:
        print "pose & euler angle not initialize"
        ret_p, pos = vrep.simxGetObjectPosition(client_id, quad_handle, -1, vrep.simx_opmode_blocking)
        ret_e, eul = vrep.simxGetObjectOrientation(client_id, quad_handle, -1, vrep.simx_opmode_blocking)
    
    pos = np.array(pos)
    eul = np.array(eul)
    next_state = np.concatenate((pos, eul, pos-last_pos, eul-last_eul, goal - pos), axis=0)
    print next_state
    if (pos[2] < floor_collision_threshold):
        print "collision with ground: ending simulation"
        stop_sim()
        return next_state, -100.0, True
    # reward 
    reward = 1.0 / (np.linalg.norm(goal - pos) + 0.1)
    return next_state, reward, timestep > max_timesteps

if __name__ == "__main__":
    global max_timesteps
    init_sim_or_die()
    num_sim = 10
    max_timesteps = 100

    for m in range(num_sim):
        print "------------- Running Simulation ", m, " / ", num_sim
        reset_sim()
        done = False
        while not done:
            s, r, done = step([5.34, 0, 0, 0])
        stop_sim()
    kill_vrep_subprocess()

