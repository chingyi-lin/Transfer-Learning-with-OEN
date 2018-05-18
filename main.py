from __future__ import division
import argparse
import os
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

#import NECAgent
import DQNAgent

#TODO: Split this into a separate agent initiation of agent and env and training
def run_agent(args):
  # Launch the graph
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session(config=config) as sess:

    # Set up training variables
    training_iters = args.training_iters
    display_step = args.display_step
    test_step = args.test_step
    test_count = args.test_count
    tests_done = 0
    test_results = []

    # Stats for display
    ep_rewards = [] ; ep_reward_last = 0
    qs = [] ; q_last = 0
    avr_ep_reward = max_ep_reward = avr_q = 0.0

    # Set precision for printing numpy arrays, useful for debugging
    #np.set_printoptions(threshold='nan', precision=3, suppress=True)
    
    mode = args.model
    
    # Create environment
    if args.env_type == 'ALE':
        from environment import ALEEnvironment
        env = ALEEnvironment(args.rom)
        if mode is None: mode = 'DQN'
        args.num_actions = env.numActions()

    elif args.env_type == 'gym':
        import gym
        try:
            import gym_vgdl #This can be found on my github if you want to use it.
        except:
            pass
        env = gym.make(args.env)
        if mode is None:
            shape = env.observation_space.shape
            if len(shape) is 3: mode = 'DQN'
            elif shape[0] is None: mode = 'object'
            else: mode = 'vanilla'
        args.num_actions = env.action_space.n #only works with discrete action spaces

    # Set agent variables
    if mode=='DQN':
        args.model = 'CNN'
        args.preprocessor = 'deepmind'
        args.obs_size = [84,84]
        args.history_len = 4
    elif mode=='image':
        args.model = 'CNN'
        args.preprocessor = 'grayscale'
        args.obs_size = list(env.observation_space.shape)[:2]
        args.history_len = 2
    elif mode=='object':
        args.model = 'object'
        args.preprocessor = 'default'
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 0
    elif mode=='vanilla':
        args.model = 'nn'
        args.preprocessor = 'default'
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 0

    # Create agent
    agent = DQNAgent.DQNAgent(sess, args)

    # Initialize all tensorflow variables
    sess.run(tf.global_variables_initializer())


    # Keep training until reach max iterations

    # Start Agent
    state = env.reset()
    agent.Reset(state)
    rewards = []
    terminal = False

    for step in tqdm(range(training_iters), ncols=80):

        #env.render()

        # Act, and add 
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)

        # Bookeeping
        rewards.append(reward)
        qs.append(value)

        if terminal:
            # Bookeeping
            ep_rewards.append(np.sum(rewards))
            rewards = []

            if step >= (tests_done)*test_step:
                tests_done += 1
                R_s = run_tests(agent, env, test_count)
                test_results.append({ 'step': step,
                          'scores': R_s,
                          'average': np.mean(R_s),
                          'max': np.max(R_s) })
                summary = { 'params': vars(args), 'tests': test_results }
                if args.save_file is not None:
                    np.save(args.save_file, summary)
                
                if args.chk_file is not None:
                    agent.Save(args.chk_file)

            # Reset agent and environment
            state = env.reset()
            agent.Reset(state)


        # Display Statistics
        if (step) % display_step == 0:
            num_eps = len(ep_rewards[ep_reward_last:])
            if num_eps is not 0:
                avr_ep_reward = np.mean(ep_rewards[ep_reward_last:])
                max_ep_reward = np.max(ep_rewards[ep_reward_last:])
                avr_q = np.mean(qs[q_last:]) ; q_last = len(qs)
                ep_reward_last = len(ep_rewards)
            tqdm.write("{}, {:>7}/{}it | {:3n} episodes,"\
                .format(time.strftime("%H:%M:%S"), step, training_iters, num_eps)
                +"q: {:4.3f}, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}, epsilon: {:4.3f}"\
                .format(avr_q, avr_ep_reward, max_ep_reward, agent.epsilon))
    
    # Continue until end of episode
    step = training_iters
    while not terminal:
        # Act, and add 
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)
        step += 1
    
    # Final test       
    R_s = run_tests(agent, env, test_count)
    test_results.append({ 'step': step,
              'scores': R_s,
              'average': np.mean(R_s),
              'max': np.max(R_s) })
    summary = { 'params': vars(args), 'tests': test_results }
    if args.save_file is not None:
        np.save(args.save_file, summary)
    
    if args.chk_file is not None:
        agent.Save(args.chk_file)
                 

def test_agent(agent, env):
    try:
        state = env.reset(train=False)
    except:
        state = env.reset()
    agent.Reset(state, train=False)
    R = 0

    terminal = False
    while not terminal:
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)
        R += reward
    return R
    
    
def run_tests(agent, env, test_count=50):
    R_s = []
    
    # Do test_count tests
    for i in tqdm(range(test_count), ncols=50,
      bar_format='Testing: |{bar}| {n_fmt}/{total_fmt}'):
        R = test_agent(agent, env)
        R_s.append(R)
        
    tqdm.write("Tests: {}".format(R_s))
    return R_s

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                       help='Gym environment to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Leave None to automatically detect')

    parser.add_argument('--seed', type=int, default=123,
                       help='Seed to initialise the agent with')

    parser.add_argument('--training_iters', type=int, default=5000000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=25000,
                       help='Number of iterations between parameter prints')
    parser.add_argument('--test_step', type=int, default=50000,
                       help='Number of iterations between tests')
    parser.add_argument('--test_count', type=int, default=5,
                       help='Number of test episodes per test')

    parser.add_argument('--learning_rate', type=float, default=0.00025,
                       help='Learning rate for TD updates')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')
    parser.add_argument('--replay_memory_size', type=int, default=100000,
                       help='Size of replay memory')
    parser.add_argument('--learn_step', type=int, default=4,
                       help='Number of steps in between learning updates')


    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial epsilon')
    parser.add_argument('--epsilon_final', type=float, default=None,
                       help='Final epsilon')
    parser.add_argument('--epsilon_anneal', type=int, default=None,
                       help='Epsilon anneal steps')

    parser.add_argument('--save_file', type=str, default=None,
                       help='Name of save file for  (leave None for no saving)')                 
    parser.add_argument('--chk_file', type=str, default=None,
                       help='Name of save file (leave None for no saving)')

    # Unused arguments
    parser.add_argument('--layer_sizes', type=str, default='64',
                       help='Hidden layer sizes for network, separate with comma (Not used)')
                       
    parser.add_argument('--n_step', type=int, default=100,
                       help='How many steps into the pass before truncating the value prediction (Not used)')
    parser.add_argument('--memory_size', type=int, default=500000,
                       help='Size of DND dictionary (Not used)')
    parser.add_argument('--num_neighbours', type=int, default=50,
                       help='Number of nearest neighbours to sample from the DND each time (Not used)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Alpha parameter for updating stored values (Not used)')
    parser.add_argument('--delta', type=float, default=0.001,
                       help='Delta parameter for thresholding closeness of neighbours (Not used)')
    parser.add_argument('--rom', type=str, default='roms/pong.bin',
                       help='Location of rom file (Not used)')

    args = parser.parse_args()

    args.env_type = 'ALE' if args.env is None else 'gym'

    if args.epsilon_final == None: args.epsilon_final = args.epsilon
    if args.epsilon_anneal == None: args.epsilon_anneal = args.training_iters

    args.layer_sizes = [int(i) for i in (args.layer_sizes.split(',') if args.layer_sizes else [])]

    arg_dict = vars(args)
    col_a_width = 20 ; col_b_width = 16
    print(' ' + '_'*(col_a_width+1+col_b_width) + ' ')
    print('|' + ' '*col_a_width + '|' + ' '*col_b_width  + '|')
    line = "|{:>" + str(col_a_width-1) + "} | {:<" + str(col_b_width-1) + "}|"
    for i in arg_dict:
        print(line.format(i, arg_dict[i]))
    print('|' + '_'*col_a_width + '|' + '_'*col_b_width  + '|')
    print('')
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    run_agent(args)

