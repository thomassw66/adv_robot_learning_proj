import vrep_main as vrep

if __name__ == "__main__":

    vrep.max_timesteps = 100
    vrep.init_sim_or_die()
    num_sim = 10

    for m in range(num_sim):
        print "------------- Running Simulation ", m, " / ", num_sim
        vrep.reset_sim()
        done = False
        while not done:
            s, r, done = vrep.step([5.34, 0, 0, 0])
        vrep.stop_sim()
    vrep.kill_vrep_subprocess()