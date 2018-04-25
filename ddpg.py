"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time
from datetime import datetime 
import vrep_main as vrep


#####################  hyper parameters  ####################

MAX_EPISODES = 200000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 1000
BATCH_SIZE = 64

RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, saved_model=None):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.a_loss = a_loss
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.td_error = td_error
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        if saved_model:
            saver = tf.train.Saver()
            saver.restore(self.sess, save_path=saved_model)
            print "successfully loaded saved model"

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def save(self, global_step):
        saver = tf.train.Saver(max_to_keep=10)
        saver.save(self.sess, 'models/my_ddpg_model', global_step=global_step)

    def restor_from_file(self, file):
        print "TODO"


    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)
            # net = tf.layers.dense(net, 64, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net = tf.layers.dense(net, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net = tf.layers.dense(net, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            # n_l1 = 64
            # w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            # w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            # b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            # net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
            net = tf.layers.dense(tf.concat([s,a], 1), 64, activation=tf.nn.relu, name='l1', trainable=trainable)
            # net = tf.layers.dense(net, 64, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net = tf.layers.dense(net, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net = tf.layers.dense(net, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, 1, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a 


if __name__ == "__main__":
    ###############################  training  ####################################

    # env = gym.make(ENV_NAME)
    # env = env.unwrapped
    # env.seed(1)

    env = vrep.VREPEnvironment(port=1234, headless=True)


    # s_dim = env.observation_space.shape[0]
    # a_dim = env.action_space.shape[0]
    # a_bound = env.action_space.high

    s_dim = 15
    a_dim = 4
    a_bound = 1

    # now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # root_logdir = "tf_logs"
    # logdir = "{}/run-{}/".format(root_logdir, now)
    global_step = 8000
    print "loading model from step ", global_step
    ddpg = DDPG(a_dim, s_dim, a_bound, saved_model='models/my_ddpg_model-{}'.format(global_step))

    # actor_loss_summary = tf.summary.scalar("Actor Loss", ddpg.a_loss)
    # tde_summary = tf.summary.scalar("TD Error", ddpg.td_error)
    # file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    # envs.max_timesteps = 100
    # vrep.init_sim_or_die()

    var = 0.5 # control exploration
    t1 = time.time()
    for i in range(global_step, MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        done = False


        # save out model 
        if i % 1000 == 0:
            ddpg.save(global_step=i)

        while not done:

            # Add exploration noise
            a = ddpg.choose_action(s)
            
            # a = np.random.normal(a, var * np.array([1.0, 0.005, 0.005, 0.005]), a.shape) 
            a = np.random.normal(a, var, a.shape)
            
            b = a.copy()
            b[0] = a[0] * 2.0 + 4.33  # scale to correct size 
            b[1:] = a[1:] * 0.04 + ( -0.02 )

            # a = np.random.normal(a, var * np.array([1.0, 0.005, 0.005, 0.005]), a.shape) # add randomness to action selection for exploration

            s_, r, done = env.step(b)
            
            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                ddpg.learn()

            s = s_
            ep_reward += r

        var *= .9995    # decay the action randomness
        print('Episode:', i, ' Reward: %.3f' % ep_reward, 'Explore: %.2f' % var, )

    # file_writer.close()
    vrep.kill_vrep_subprocess()
    print('Running time: ', time.time() - t1)
