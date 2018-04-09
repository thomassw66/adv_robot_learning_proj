import time 
import sys 
import os

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

def init_vrep_connection_or_die():
    global client_id
    global quad_handle 
    # --------------------------------------------------------
    vrep.simxFinish(-1) # close any opened conenctions 
    client_id = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if client_id == -1:
        print 'Failed to connect to remote API server'
        sys.exit('could not connect')
    
    print ('Connection to Remote Api Server: Established')
    # -------------------------------------------------------
    res, quad_handle = vrep.simxGetObjectHandle(client_id, "Quadricopter_base", vrep.simx_opmode_blocking)
    if res != vrep.simx_return_ok:
        print 'Could not GetObjectHandle: Quadricopter_base'
        sys.exit('Failed to grab Quadricopter_base handle')
    print quad_handle
    print ('grabbed Quadricopter_base handle ')

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

stream_flag = False

def get_quad_pose():
    
    """
    global stream_flag
    if stream_flag==False: 
        opmode = vrep.simx_opmode_streaming
        stream_flag = True
    else: 
        opmode = vrep.simx_opmode_buffer
    """
    opmode = vrep.simx_opmode_blocking
    # -1 specifies pose relative to global coordinate frame
    ret, pos = vrep.simxGetObjectPosition(client_id, quad_handle, -1, opmode)
    ret, eul = vrep.simxGetObjectOrientation(client_id, quad_handle, -1, opmode)

    print "position: ", pos
    print "orientation: ", eul

def pos_update():
    global quad_handle 
    print quad_handle
    while vrep.simxGetConnectionId(client_id) != -1: # while we are still connected to the server 
        ret, pos = vrep.simxGetObjectPosition(client_id, quad_handle, vrep.sim_handle_parent, vrep.simx_opmode_blocking)
        if (ret == vrep.simx_return_ok):
            print pos
        else:
            print 'position not initialized'
        time.sleep(0.3)
    

if __name__ == "__main__":
    init_vrep_connection_or_die()
    #start_state_estimation()
    #start_controller()

    ret, pos = vrep.simxGetObjectPosition(
            client_id, 
            quad_handle, 
            -1, 
            vrep.simx_opmode_streaming)
    ret, eul = vrep.simxGetObjectOrientation(
            client_id, 
            quad_handle, 
            -1, 
            vrep.simx_opmode_streaming)

    while True:
        # get_quad_pose()
        setpoint_([5.32, 0, 0, 0.2])

        ret_p, pos = vrep.simxGetObjectPosition(
                client_id, 
                quad_handle, 
                -1, 
                vrep.simx_opmode_streaming)
        ret_e, eul = vrep.simxGetObjectOrientation(
                client_id, 
                quad_handle, 
                -1, 
                vrep.simx_opmode_streaming)

        if ret_p == vrep.simx_return_ok:
            print 'pos: ', pos
        else:
            print 'pose not initialized'

        if ret_e == vrep.simx_return_ok:
            print 'eul: ', eul
        else:
            print 'orientation not initialized'

        time.sleep(0.1)

