from __future__ import print_function

import argparse
import configparser
import csv
import logging
import os
import sys
from datetime import datetime

import gym
import gym.wrappers as wrappers
import numpy as np
from Agent.RACAgent import RACAgent


class RAC(object):
    def __init__(self):
        self.__initialise_agents()

    def __initialise_agents(self):
        self.agent = RACAgent(props.getint('feature', 'dimension'),
                              props.getint('policy', 'num_actions'),
                              props.getint('state', 'state_length'),
                              props.getint('state', 'feature_size'))

    def execute_algorithm(self):
        max_episodes = props.getint('train', 'max_episodes')
        max_steps = props.getint('train', 'max_steps')
        step_size = props.getint('train', 'step_size')
        update_count = 0
        for i in range(max_episodes):
            t_start = datetime.now()
            steps = 0
            state = env.reset()
            terminal_reached = False
            while not terminal_reached and steps < max_steps:
                # Predict action
                state_feature = self.agent.feature.phi(state)
                action, distribution = self.agent.policy.get_action(state_feature)
                # take action and observe reward
                next_state, reward, done, info = env.step(action)

                for x in range(step_size - 1):
                    if done:
                        terminal_reached = True
                        break
                    next_state, reward2, done, info = env.step(action)
                    reward += reward2

                if done:
                    terminal_reached = True

                steps += 1
                self.agent.update_parameters(state, action, reward, next_state)
                update_count += 1
                state = next_state
                # test agent
                test_agent(self.agent, update_count)
                logger.debug("Completed Iteration. Time taken: %f", (datetime.now() - t_start).total_seconds())


def test_agent(agent, episode_count):
    env_test = gym.make(args.env_id)
    if display_game:
        outdir = 'videos/tmp/neat-data/{0}-{1}'.format(env_test.spec.id, str(datetime.now()))
        env_test = wrappers.Monitor(env_test, directory=outdir, force=True)


    logger.debug("Generating best agent result: %d", episode_count)
    t_start = datetime.now()

    test_episodes = props.getint('test', 'test_episodes')
    step_size = props.getint('test', 'step_size')

    avg_steps = []
    avg_rewards = []

    for i in range(test_episodes):
        state = env_test.reset()
        terminal_reached = False
        steps = 0
        rewards = 0
        while not terminal_reached:
            if display_game:
                env.render()

            # Predict action
            state_feature = agent.feature.phi(state)
            action, distribution = agent.policy.get_action(state_feature)
            # take action and observe reward
            next_state, reward, done, info = env_test.step(action)

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                next_state, reward2, done, info = env_test.step(action)
                reward += reward2

            steps += 1
            rewards += reward
            state = next_state
            if done:
                terminal_reached = True
        avg_steps.append(steps)
        avg_rewards.append(rewards)

    average_steps_per_episode = np.sum(avg_steps) / len(avg_steps)
    average_rewards_per_episode = np.sum(avg_rewards) / len(avg_rewards)

    # save this to file along with the generation number
    entry = [episode_count, average_steps_per_episode, average_rewards_per_episode]
    with open(r'agent_evaluation-{0}.csv'.format(time), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(entry)

    logger.debug("Finished: evaluating best agent. Time taken: %f", (datetime.now() - t_start).total_seconds())
    env_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    parser.add_argument('display', nargs='?', default='false', help='Show display of game. true or false')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    logging.basicConfig(filename='log/debug-{0}.log'.format(time),
                        level=logging.DEBUG, format='[%(asctime)s] %(message)s')
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.DEBUG)

    env = gym.make(args.env_id)

    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)

    # Load the properties file
    local_dir = os.path.dirname(__file__)
    logger.debug("Loading Properties")
    props = configparser.ConfigParser()
    prop_path = os.path.join(local_dir, 'properties/{0}/neatem_properties.ini'.format(env.spec.id))
    props.read(prop_path)
    logger.debug("Finished: Loading Properties")

    agent = RAC()

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    display_game = True if args.display == 'true' else False



    try:
        agent.execute_algorithm()
    except KeyboardInterrupt:
        logger.debug("User break.")

    env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
