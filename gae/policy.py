import tensorflow as tf
import numpy as numpy

class Policy(object):

    def __init__(self, env_spec, internal_dim, fixed_std=True):
        self.env_spec = env_spec
        self.internal_dim = internal_dim 
        self.fixed_std = fixed_std
        self.matrix_init = tf.truncated_normal_initializer(stddev=0.01)
        self.vector_init = tf.constant_initializer(0.0)

    @property
    def input_dim(self):
        return self.env_spec.total_obs_dim
    
    @property
    def output_dim(self):
        return self.env_spec.total_sampling_act_dim

    def core(self, obs):
        batch_size = tf.shape(obs[0])[0]
        b = tf.get_variable('input_bias', [self.input_dim])
