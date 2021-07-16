import csv
import math
from statistics import mean
import pandas as pd
import time
from datetime import timedelta
import numpy as np
from tqdm import tqdm

column_names = ["action","x", "y", "x'", "y'", "angle", "angular speed",
                "first leg contact", "second leg contact"]


class Experiment:
    def __init__(self, environment, agent=None, load_models=False, config=None):
        self.counter = 0
        self.test = 0
        self.config = config
        self.env = environment
        self.agent = agent
        self.best_score = None
        self.best_reward = None
        self.best_score_episode = -1
        self.best_score_length = -1
        self.total_steps = 0
        self.action_history = []
        self.score_history = []
        self.episode_duration_list = []
        self.length_list = []
        self.grad_updates_durations = []
        self.test_length_list = []
        self.test_score_history = []
        self.test_episode_duration_list = []
        self.discrete = config['SAC']['discrete'] if 'SAC' in config.keys() else None
        self.second_human = config['game']['second_human'] if 'game' in config.keys() else None
        self.duration_pause_total = 0
        if load_models:
            self.agent.load_models()
        self.df = pd.DataFrame(columns=column_names)
        self.df_test = pd.DataFrame(columns=column_names)
        self.max_episodes = None
        self.max_timesteps = None
        self.avg_grad_updates_duration = 0
        self.human_action = 0
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.agent_action = None
        self.total_timesteps = None
        self.max_timesteps_per_game = None
        self.save_models = True
        self.game = None
        self.test_max_timesteps = self.config['Experiment']['test_loop']['max_timesteps'] if 'test_loop' in config['Experiment'].keys() else None
        self.test_max_episodes = self.config['Experiment']['test_loop']['max_games'] if 'test_loop' in config['Experiment'].keys() else None
        self.update_cycles = None

        # Experiment 1 loop

    def loop_1(self):
        # Experiment 1 loop
        flag = True
        current_timestep = 0
        running_reward = 0
        avg_length = 0

        self.best_score = -100 - 1 * self.config['Experiment']['loop_1']['max_timesteps']
        self.best_reward = self.best_score
        self.max_episodes = self.config['Experiment']['loop_1']['max_episodes']
        self.max_timesteps = self.config['Experiment']['loop_1']['max_timesteps']

        # self.test_agent(goal, 1)
        # print("Continue Training.")

        for i_episode in range(1, self.max_episodes + 1):
            observation = self.env.reset()
            timedout = False
            episode_reward = 0
            test_offline_score = 0
            start = time.time()

            #actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            self.save_models = True
            for timestep in range(1, self.max_timesteps + 1):
                self.total_steps += 1
                current_timestep += 1

                # compute agent's action
                if not self.second_human:
                    randomness_threshold = self.config['Experiment']['loop_1']['stop_random_agent']
                    randomness_critirion = i_episode
                    flag = self.compute_agent_action(observation, randomness_critirion, randomness_threshold, flag)
                
                
                # get final action pair
                if not self.config['game']['agent_only']:
                    if self.human_action:
                        self.env.step(self.human_action)
                    action = self.agent_action
                else:
                    action = self.agent_action
                
                if timestep == self.max_timesteps:
                    timedout = True

                # Environment step
                
                observation_, reward, done, _ = self.env.step(action)
                
                #, timedout, goal,
                                                           #self.config['Experiment']['loop_1']['action_duration'])
                # add experience to buffer
                # interaction = [observation, self.agent_action, reward, observation_, done]
                if not self.config['game']['agent_only']:
                    if action == 2:
                        action = 1 # fix indexing issue
                    interaction = [observation, action, reward, observation_, done]
                    self.save_experience(interaction)
                else:
                    interaction = [observation, action, reward, observation_, done]
                    self.save_experience(interaction)

                running_reward += reward
                episode_reward += reward
                test_offline_score += -1 if not done else 0

                # online train
                if not self.config['game']['test_model'] and not self.second_human:
                    if self.config['Experiment']['online_updates'] and i_episode >= self.config['Experiment']['loop_1'][
                        'start_training_step_on_episode']:
                        if self.discrete:
                            self.agent.learn()
                            self.agent.soft_update_target()

                observation = observation_
                new_row = {'action': action, "x": observation[0],"y": observation[1], "x'": observation[2], "y'": observation[3],
                           "angle": observation[4], "angular speed": observation[5], "first leg contact": observation[6],
                           "second leg contact": observation[7]}
                # append row to the dataframe
                self.df = self.df.append(new_row, ignore_index=True)
                # if total_steps >= start_training_step and total_steps % sac.target_update_interval == 0:
                #     sac.soft_update_target()

                window_still_open = self.env.render()
                if window_still_open==False: return False

                if done:
                    break

                while self.human_sets_pause:
                    self.env.render()
                    time.sleep(0.1)
                
                time.sleep(0.1)

            end = time.time()
            if self.best_reward < episode_reward:
                self.best_reward = episode_reward
            print('Total episode reward: ', episode_reward)
            self.duration_pause_total += duration_pause
            episode_duration = end - start - duration_pause

            self.episode_duration_list.append(episode_duration)
            self.score_history.append(episode_reward)

            log_interval = self.config['Experiment']['loop_1']['log_interval']
            avg_ep_duration = np.mean(self.episode_duration_list[-log_interval:])
            avg_score = np.mean(self.score_history[-log_interval:])

            # best score logging
            # self.save_best_model(avg_score, i_episode, current_timestep)

            self.length_list.append(current_timestep)
            avg_length += current_timestep

            # if not self.config['Experiment']['online_updates']:
            #     self.test_score_history.append(self.config['Experiment']['test_loop']['max_score'] + test_offline_score)
            #     self.test_episode_duration_list.append(episode_duration)
            #     self.test_length_list.append(current_timestep)

            # off policy learning
            if not self.config['game']['test_model'] and i_episode >= self.config['Experiment']['loop_1'][
                'start_training_step_on_episode']:
                if i_episode % self.agent.update_interval == 0:
                    self.updates_scheduler()
                    if self.update_cycles > 0:
                        grad_updates_duration = self.grad_updates(self.update_cycles)
                        self.grad_updates_durations.append(grad_updates_duration)

                        # save the models after each grad update
                        self.agent.save_models()

                    # Test trials
                    if i_episode % self.config['Experiment']['test_interval'] == 0 and self.test_max_episodes > 0:
                        # self.test_agent(goal)
                        self.test_agent()
                        print("Continue Training.")

            # logging
            if self.config["game"]["verbose"]:
                if not self.config['game']['test_model']:
                    running_reward, avg_length = self.print_logs(i_episode, running_reward, avg_length, log_interval,
                                                                 avg_ep_duration)
                current_timestep = 0
        update_cycles = math.ceil(
            self.config['Experiment']['loop_1']['total_update_cycles'])
        if not self.second_human and update_cycles > 0:
            try:
                self.avg_grad_updates_duration = mean(self.grad_updates_durations)
            except:
                print("Exception when calc grad_updates_durations")
    
    def key_press(self, key, mod):
        if key==0xff0d: self.human_wants_restart = True
        if key==32: self.human_sets_pause = not self.human_sets_pause
        
        if key == 65363: a = 1
        #elif key == 65364: a = 2
        elif key == 65361: a = 3
        else: a = 0
        
        if a <= 0 or a >= self.env.action_space.n: return
        self.human_action = a
        
    def key_release(self, key, mod):        
        if key == 65363: a = 1
        #elif key == 65364: a = 2
        elif key == 65361: a = 3
        else: a = 0
        
        if a <= 0 or a >= self.env.action_space.n: return
        if self.human_action == a:
            self.human_action = 0    

    def save_info(self, chkpt_dir, experiment_duration, total_games):
        info = {}
        #info['goal'] = goal
        info['experiment_duration'] = experiment_duration
        info['best_score'] = self.best_score
        info['best_score_episode'] = self.best_score_episode
        info['best_reward'] = self.best_reward
        info['best_score_length'] = self.best_score_length
        info['total_steps'] = self.total_steps
        info['total_games'] = total_games
        #info['fps'] = self.env.fps
        info['avg_grad_updates_duration'] = self.avg_grad_updates_duration
        w = csv.writer(open(chkpt_dir + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])

    def get_action_pair(self):
        if self.second_human:
            action = self.human_actions
        else:
            if self.config['game']['agent_only']:
                action = self.get_agent_only_action()
            else:
                action = [self.agent_action, self.human_action]
        self.action_history.append(action)
        return action

    def save_experience(self, interaction):
        observation, agent_action, reward, observation_, done = interaction
        if not self.second_human:
            if self.discrete:
                self.agent.memory.add(observation, agent_action, reward, observation_, done)
            else:
                self.agent.remember(observation, agent_action, reward, observation_, done)

    def save_best_model(self, avg_score, game, current_timestep):
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_score_episode = game
            self.best_score_length = current_timestep
            if not self.config['game']['test_model'] and self.save_models and not self.second_human:
                self.agent.save_models()

    def grad_updates(self, update_cycles=None):
        start_grad_updates = time.time()
        end_grad_updates = 0
        if not self.second_human:
            print("Performing {} updates".format(update_cycles))
            for _ in tqdm(range(update_cycles)):
                if self.discrete:
                    self.agent.learn()
                    self.agent.soft_update_target()
                else:
                    self.agent.learn()
            end_grad_updates = time.time()

        return end_grad_updates - start_grad_updates

    def print_logs(self, game, running_reward, avg_length, log_interval, avg_ep_duration):
        if game % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            log_reward = int((running_reward / log_interval))

            print(
                'Episode {}\tTotal timesteps {}\tavg length: {}\tTotal reward(last {} episodes): {}\tavg '
                'episode duration: {}'.format(game, self.total_steps, avg_length,
                                              log_interval,
                                              log_reward,
                                              timedelta(
                                                  seconds=avg_ep_duration)))
            running_reward = 0
            avg_length = 0
        return running_reward, avg_length

    def test_print_logs(self, avg_score, avg_length, best_score, duration):
        print(
            'Avg Score: {}\tAvg length: {}\tBest Score: {}\tTest duration: {}'.format(avg_score,
                                                                                      avg_length, best_score,
                                                                                      timedelta(seconds=duration)))

    def compute_agent_action(self, observation, randomness_critirion=None, randomness_threshold=None, flag=True):
        if self.discrete:
            if randomness_critirion is not None and randomness_threshold is not None \
                    and randomness_critirion <= randomness_threshold:
                # Pure exploration
                if self.config['game']['agent_only']:
                    #self.agent_action = np.random.randint(pow(2, self.env.action_space.n))
                    self.agent_action = np.random.randint(self.env.action_space.n)
                else:
                    self.agent_action = np.random.choice([0,2])
                self.save_models = False
                if flag:
                    print("Using Random Agent")
                    flag = False
            else:  # Explore with actions_prob
                self.save_models = True
                self.agent_action = self.agent.actor.sample_act(observation)
                if not self.config['game']['agent_only']:
                    if self.agent_action == 1:
                            self.agent_action = 2 # adjust indexing case
                if not flag:
                    print("Using SAC Agent")
                    flag = True
        #else:
            #self.save_models = True
            #self.agent_action = self.agent.choose_action(observation)
        return flag

    def test_agent(self, randomness_critirion=None):
        # test loop
        current_timestep = 0
        self.test += 1
        print('Test {}'.format(self.test))
        best_score = 0
        for game in range(1, self.test_max_episodes + 1):
            observation = self.env.reset()
            timedout = False
            episode_reward = 0
            start = time.time()

            actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            for timestep in range(1, self.test_max_timesteps + 1):
                current_timestep += 1
                # compute agent's action
                randomness_threshold = self.config['Experiment']['loop_2']['start_training_step_on_timestep']
                self.compute_agent_action(observation, randomness_critirion, randomness_threshold)
                
                # get final action pair
                #action = self.get_action_pair()
                if not self.config['game']['agent_only']:
                    if self.human_action:
                        self.env.step(self.human_action)
                    action = self.agent_action
                    if action == 1:
                        action = 2 # adjust indexing case
                else:
                    action = self.agent_action

                if timestep == self.test_max_timesteps:
                    timedout = True

                # Environment step
                observation_, _, done, _ = self.env.step(action)
                #, timedout, goal,
                                                      #self.config['Experiment']['test_loop']['action_duration'])

                observation = observation_
                new_row = {'action': action, "x": observation[0],"y": observation[1], "x'": observation[2], "y'": observation[3],
                           "angle": observation[4], "angular speed": observation[5], "first leg contact": observation[6],
                           "second leg contact": observation[7]}
                # append row to the dataframe
                self.df_test = self.df_test.append(new_row, ignore_index=True)

                episode_reward += -1

                window_still_open = self.env.render()
                if window_still_open==False: return False

                if done:
                    break

                while self.human_sets_pause:
                    self.env.render()
                    time.sleep(0.1)

                time.sleep(0.1)

            end = time.time()

            self.duration_pause_total += duration_pause
            episode_duration = end - start - duration_pause
            episode_score = episode_reward

            self.test_episode_duration_list.append(episode_duration)
            self.test_score_history.append(episode_score)
            self.test_length_list.append(current_timestep)
            best_score = episode_score if episode_score > best_score else best_score

            current_timestep = 0

        # logging
        self.test_print_logs(mean(self.test_score_history[-10:]), mean(self.test_length_list[-10:]), best_score,
                             sum(self.test_episode_duration_list[-10:]))

    def get_agent_only_action(self):
        # up: 0, down:1, left:2, right:3, upleft:4, upright:5, downleft: 6, downright:7
        if self.agent_action == 0:
            return 0
        elif self.agent_action == 1:
            return 1
        elif self.agent_action == 2:
            return 2
        elif self.agent_action == 3:
            return 3
        else:
            print("Invalid agent action")

    def test_loop(self):
        # test loop
        current_timestep = 0
        self.test += 1
        print('Test {}'.format(self.test))
        #goals = [left_down, right_down, left_up, ]
        for game in range(1, self.test_max_episodes + 1):
            # randomly choose a goal
            #current_goal = random.choice(goals)

            observation = self.env.reset()
            timedout = False
            episode_reward = 0
            start = time.time()

            actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            self.save_models = False
            for timestep in range(1, self.test_max_timesteps + 1):
                self.total_steps += 1
                current_timestep += 1

                # compute agent's action
                self.compute_agent_action(observation)
                
                # get final action pair
                if not self.config['game']['agent_only']:
                    if self.human_action:
                        self.env.step(self.human_action)
                        action = self.agent_action
                    else: 
                        action = self.agent_action
                        if action == 1:
                            action = 2 # adjust indexing case
                else:
                    action = self.agent_action
                

                if timestep == self.max_timesteps:
                    timedout = True

                # Environment step
                observation_, reward, done, _ = self.env.step(action)
                #, timedout, current_goal,
                                                           #self.config['Experiment']['test_loop']['action_duration'])

                observation = observation_
                
                new_row = {'action': action, "x": observation[0],"y": observation[1], "x'": observation[2], "y'": observation[3],
                           "angle": observation[4], "angular speed": observation[5], "first leg contact": observation[6],
                           "second leg contact": observation[7]}
                # append row to the dataframe
                self.df_test = self.df_test.append(new_row, ignore_index=True)

                episode_reward += reward

                window_still_open = self.env.render()
                if window_still_open==False: return False

                if done:
                    break

                while self.human_sets_pause:
                    self.env.render()
                    time.sleep(0.1)
                
                time.sleep(0.1)

            end = time.time()

            self.duration_pause_total += duration_pause
            episode_duration = end - start - duration_pause

            print('Total reward: ', episode_reward)
            
            self.test_episode_duration_list.append(episode_duration)
            self.test_score_history.append(episode_reward)
            self.test_length_list.append(current_timestep)

            # logging
            # self.test_print_logs(game, episode_reward, current_timestep, episode_duration)

            current_timestep = 0

    def updates_scheduler(self):
        update_list = [22000, 1000, 1000, 1000, 1000, 1000, 1000]
        total_update_cycles = self.config['Experiment']['loop_1']['total_update_cycles']
        online_updates = 0
        if self.config['Experiment']['online_updates']:
            online_updates = self.max_timesteps * (
                    self.max_episodes - self.config['Experiment']['loop_1']['start_training_step_on_episode'])

        if self.update_cycles is None:
            self.update_cycles = total_update_cycles - online_updates

        if self.config['Experiment']['scheduling'] == "descending":
            self.counter += 1
            if not (math.ceil(self.max_episodes / self.agent.update_interval) == self.counter):
                self.update_cycles /= 2

        elif self.config['Experiment']['scheduling'] == "big_first":
            if self.config['Experiment']['online_updates']:
                if self.counter == 1:
                    self.update_cycles = update_list[self.counter]
                else:
                    self.update_cycles = 0
            else:
                self.update_cycles = update_list[self.counter]

            self.counter += 1

        else:
            self.update_cycles = (total_update_cycles - online_updates) / math.ceil(
                self.max_episodes / self.agent.update_interval)

        self.update_cycles = math.ceil(self.update_cycles)
