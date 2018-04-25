import tensorflow as tf
import vrep_main as vrep 
from ddpg import DDPG

# init = tf.global_variables_initializer()
# sess = tf.Session()

if __name__ == "__main__":
    env = vrep.VREPEnvironment(port=19997, headless=False)
    agent = DDPG(4, 15, 1, saved_model='models/my_ddpg_model-15000')
    
    n_max_steps = 100
    n_iterations = 4
    for i in xrange(n_iterations):
        state = env.reset()
        episode_reward = 0.0
        print env.goal
        for step in xrange(n_max_steps):
            a = agent.choose_action(state)
            b=a.copy()
            b[0] = a[0] * 2.0 + 4.33  # scale to correct size 
            b[1:] = a[1:] * 0.04 + ( -0.02 )
            state, reward, done = env.step(b)
            episode_reward += reward
            print state
            if done: break
        print "Finished episode: reward = ", episode_reward
    env.close()
    print "tada!"