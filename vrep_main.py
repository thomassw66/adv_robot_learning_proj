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

class VREPSimulatorConnection(object):
    def __init__(self, path_to_vrep_dir='/home/thomas/V-REP/',host='127.0.0.1', port=19991, headless=False):
        self.port = port 
        self.host = host
        self.vrep_dir = path_to_vrep_dir
        self.vrep_path = os.path.join(path_to_vrep_dir, 'vrep')
        self.process = None 
        self.is_headless = headless

    def kill(self):
        p = psutil.Process(self.process.pid)
        for proc in p.children(recursive=True):
            proc.kill()
        p.kill()

    def start_subprocess(self, scene="vrep_scenes/modified_quad.ttt"):

        if not os.path.exists(self.vrep_path):
            print "vrep is not at \"{0}\". cannot start simulator".format(self.vrep_dir)
            sys.exit("exiting. failed to start simulator process")

        self.scene_path = os.path.abspath(scene) 
        if not os.path.exists(self.vrep_path):
            print "cannot find scene file at {0}".format(self.scene_path)
            sys.exit("exiting. failed to find scene file")

        mod_env = os.environ.copy()
        if "LD_LIBRARY_PATH" in mod_env: # so vrep can find its link path 
            mod_env["LD_LIBRARY_PATH"] = self.vrep_dir + ":" + mod_env["LD_LIBRARY_PATH"]
        else:
            mod_env["LD_LIBRARY_PATH"] = self.vrep_dir
        args = []
        if self.is_headless: # headless mode for faster sim 
            args = [self.vrep_path, "-gREMOTEAPISERVERSERVICE_{0}_FALSE_TRUE".format(self.port), "-h", self.scene_path] 
        else:                # normal mode 
            args = [self.vrep_path, "-gREMOTEAPISERVERSERVICE_{0}_FALSE_TRUE".format(self.port), self.scene_path]
        self.process = subprocess.Popen(args=args, env=mod_env)
        # this is about how long it takes to spin up vrep (on my mbp)
        # the only real reason to do this is to give the remoteApi time
        # so we can connect to it with python
        time.sleep(7.0)
        print "--------- Finished Waiting for VREP --------"

    def connect(self):
        # vrep.simxFinish(-1) # close any opened conenctions (saw someone else doing it)
        self.client_id = vrep.simxStart(self.host, self.port, True, True, 5000, 5)
        if self.client_id == -1:
            print 'Failed to connect to remote API server @ {0}:{1}'.format(self.host, self.port)
            sys.exit('could not connect')
        print ('Connection to Remote Api Server: Established')


class QuadricopterSimulation(object):
    def __init__(self, connection=None):
        self.connection = connection 
        self.actuate_signal = "objectIdAndForce"
        res, qh = vrep.simxGetObjectHandle(self.client_id, "Quadricopter_base", vrep.simx_opmode_blocking)
        if res != vrep.simx_return_ok:
            print "could not get object handle for quadricopter_base"
            sys.exit("exiting: failed to grab quadcopter_base handle");
        self.quad_handle = qh 
        print "getObjectHandle succeeded for Quadricopter_base"
        
    @property
    def client_id(self):
        return self.connection.client_id

    @property
    def state(self):
        ret_p, pos = vrep.simxGetObjectPosition(self.client_id, self.quad_handle, -1, vrep.simx_opmode_blocking)
        ret_e, eul = vrep.simxGetObjectOrientation(self.client_id, self.quad_handle, -1, vrep.simx_opmode_blocking)
        while ret_p != vrep.simx_return_ok or ret_e != vrep.simx_return_ok:
            print "pose & euler angle not ready"
            ret_p, pos = vrep.simxGetObjectPosition(self.client_id, self.quad_handle, -1, vrep.simx_opmode_blocking)
            ret_e, eul = vrep.simxGetObjectOrientation(self.client_id, self.quad_handle, -1, vrep.simx_opmode_blocking)
        pos = np.array(pos)
        eul = np.array(eul)
        return pos, eul
    
    # Takes a float in R^4 and send actions to the simulator in the for of [thrust moments(r,p,y)]
    def actuate(self, action):
        data = vrep.simxPackFloats(action)
        vrep.simxSetStringSignal(self.client_id, self.actuate_signal, data, vrep.simx_opmode_blocking)


class VREPEnvironment(object):
    def __init__(self, port=None, headless=False):
        self.connection = VREPSimulatorConnection(port=port, headless=headless)
        self.connection.start_subprocess()
        self.connection.connect()
        self.quadcopter_sim = QuadricopterSimulation(connection=self.connection)

        self.env_bound = ((-5.0, -5.0, 0.09), (5.0, 5.0, 5.0))
        self.floor_collision_threshold = 0.09
        self.goal = None 
        self.timestep = 0
        self.pos = np.array([0.0, 0.0, 0.0])
        self.eul = np.array([0.0, 0.0, 0.0])
        self.last_pos = None 
        self.last_eul = None

    def generate_new_goal(self):
        # return np.array([2.0, 2.0, 1.0])
        return np.concatenate((np.random.rand(2) * 10 - 5, np.random.rand(1) * 5), axis=0) 

    def reset(self):
        _ = vrep.simxStopSimulation(self.connection.client_id, vrep.simx_opmode_blocking)
        time.sleep(0.1)
        _ = vrep.simxStartSimulation(self.connection.client_id, vrep.simx_opmode_blocking)
        self.goal = self.generate_new_goal()
        self.timestep = 0
        self.pos, self.eul = self.quadcopter_sim.state 
        return np.concatenate((self.pos, self.eul, [0.0,0.0,0.0], [0.0,0.0,0.0], self.goal-self.pos), axis=0)

    def check_point_in_bounds(self, p):    
        low, up= self.env_bound
        return p[0] > low[0] and p[0] < up[0] and \
            p[1] > low[1] and p[1] < up[1] and \
            p[2] > low[2] and p[2] < up[2]

    def step(self, action):
        self.timestep += 1
        self.last_eul = self.eul
        self.last_pos = self.pos 
        self.quadcopter_sim.actuate(action)
        self.pos, self.eul = self.quadcopter_sim.state 
        dr = self.pos - self.last_pos 
        dw = self. eul - self.last_eul
        relative_goal = self.goal - self.pos 
        dist_to_goal = np.linalg.norm(relative_goal)
        next_state = np.concatenate((self.pos, self.eul, dr, dw, relative_goal), axis=0)
        if not self.check_point_in_bounds(self.pos):
            return next_state, -100.0, True
        reward = np.linalg.norm(np.dot(dr, relative_goal)) - np.linalg.norm(dr)
        return next_state, reward, False # timestep > max_timesteps

    def close(self):
        self.connection.kill()




if __name__ == "__main__":
    env = VREPEnvironment(port=12345, headless=True)
    n_max_steps = 100
    n_iterations = 10
    for i in xrange(n_iterations):
        env.reset()
        print env.goal
        for step in xrange(n_max_steps):
            state, reward, done = env.step([5.34, 0.0, 0.0, 0.0])
            print state
            if done: break
    env.close()
    print "tada!"
