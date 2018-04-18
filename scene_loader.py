import sys 
import vrep 
import os
import ctypes as ct 
import time

address = '127.0.0.1'
port = 19997
param_options = 1#ct.c_ubyte(0x0001)

if __name__ == "__main__":
    vrep.simxFinish(-1)
    client_id = vrep.simxStart(connectionAddress='127.0.0.1', connectionPort=19999, waitUntilConnected=True, doNotReconnectOnceDisconnected=True, timeOutInMs=5000, commThreadCycleInMs=5)
    if client_id == -1:
        print "Failed to connect to V-REP simulator" 
        sys.exit("failed to connect")

    try:
        ret, _ = vrep.simxLoadScene(client_id, 'vrep_scenes/modified_quad.ttt', 0xFF, vrep.simx_opmode_blocking)
    except:
        print "it failed but it worked. Why? we don't know"