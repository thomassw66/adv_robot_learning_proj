

def run_policy_gradient_algorithm(env, agent, usercfg=None, callback=None):
    print "not implemented"

def rollout(env, agent, timestep_limit):
    """ simulate the env and agent for timestep_limit steps """
    ob = env.reset()
    terminated = False 

    data = defaultdict