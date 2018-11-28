import numpy as np
import cv2
import tensorflow as tf

from ops import linear
from gym_utils.replay_memory import ReplayMemory
import utils
from ModelBasedPolicy import ModelBasedPolicy

# TODO:
# build graph
# tf placeholder
# have initial random dataset


class ModelBasedAgent(object):
    def __init__(self, 
                 session, 
                 args,
                 env, 
                 mpc_horizon=15,
                 max_rollout_length=500, 
                 num_init_random_rollouts=10,
                 num_onplicy_iters=10, 
                 num_onpolicy_rollouts=10, 
                 num_random_action_selection=4096,
                 nn_layers=1,
                 training_epochs=60,
                 training_batch_size=512
                 ):
        self.name = "ModelBasedAgent"

        # Environment details
        self.env = env
        self.render = args.render
        self.obs_size = args.obs_size
        self.n_actions = args.num_actions
        # TODO: CY's question - what is viewer?
        self.viewer = None
        self.seed = args.seed

        # Reinforcement learning parameters
        self.n_steps = args.n_step

        self._max_rollout_length = max_rollout_length
        self._num_onpolicy_iters = num_onplicy_iters
        self._num_onpolicy_rollouts = num_onpolicy_rollouts
        self._training_epochs = training_epochs
        self._training_batch_size = training_batch_size

        from networks import object_embedding_network2
        self.model = object_embedding_network2

        def wrap_env(env, wrappers):
            for wrapper in wrappers:
                env = wrapper(env)
            return env
        
        wrapped_env = lambda: wrap_env(env_cons(), wrappers)

        print('Gathering random dataset')
        self._random_dataset = self._gather_rollouts(utils.RandomPolicy(self.n_actions, self.env), num_init_random_rollouts)
        
        print('Creating policy')
        self._policy = ModelBasedPolicy(self.env,
                                        self._random_dataset,
                                        horizon=mpc_horizon,
                                        num_random_action_selection=num_random_action_selection)

    def Reset(self):
        # reinitiate graphs and reconstruct a random rollouts?
        self._random_dataset = self._gather_rollouts(utils.RandomPolicy(self.n_actions, self.env), num_init_random_rollouts)
        self._policy = ModelBasedPolicy(self.env,
                                        self._random_dataset,
                                        horizon=mpc_horizon,
                                        num_random_action_selection=num_random_action_selection)
    
    def Save(self, save_dir):
        # Save model to file
        self.saver.save(self.session, save_dir + '/model.ckpt')
        
    def Load(self, save_dir):
        # Load model from file
        ckpt = tf.train.get_checkpoint_state(save_dir)
        print("Loading model from {}".format(ckpt.model_checkpoint_path))
        self.saver.restore(self.session, ckpt.model_checkpoint_path)

        # TODO: need to initialise network
    
    def MBRLTraining(self):
        state = env.reset()
        rewards = []
        for itr in range(self._num_onpolicy_iters + 1):
            print('Iteration {0}'.format(itr))
            
            ### PROBLEM 3
            ### YOUR CODE HERE
            print('Training policy...')
            # raise NotImplementedError
            self._train_policy(self._random_dataset)

            ### PROBLEM 3
            ### YOUR CODE HERE
            print('Gathering rollouts...')
            # raise NotImplementedError
            new_dataset = self._gather_rollouts(self._policy, self._num_onpolicy_rollouts)

            ### PROBLEM 3
            ### YOUR CODE HERE
            print('Appending dataset...')
            # raise NotImplementedError
            dataset.append(new_dataset)

    def _gather_rollouts(self, policy, num_rollouts):
        dataset = utils.Dataset()
        for _ in range(num_rollouts):
            state = self.env.reset()
            done = False
            t = 0
            while not done:
                if self.render:
                    self.env.render()
                action = policy.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                done = done or (t >= self._max_rollout_length)
                dataset.add(state, action, next_state, reward, done)

                state = next_state
                t += 1

    def _train_policy(self, dataset):
        """
        Train the model-based policy

        implementation details:
            (a) Train for self._training_epochs number of epochs
            (b) The dataset.random_iterator(...)  method will iterate through the dataset once in a random order
            (c) Use self._training_batch_size for iterating through the dataset
            (d) Keep track of the loss values by appending them to the losses array
        """

        losses = []
        for i in range(self._training_epochs):
            loss_epoch = []
            for (states, actions, next_states, _, _) in (dataset.random_iterator(self._training_batch_size)):
                loss = self._policy.train_step(states, actions, next_states)
                loss_epoch.append(loss)
            losses.append(np.mean(loss_epoch))

        # logger.record_tabular('TrainingLossStart', losses[0])
        # logger.record_tabular('TrainingLossFinal', losses[-1])

        timeit.stop('train policy')

        return dataset
