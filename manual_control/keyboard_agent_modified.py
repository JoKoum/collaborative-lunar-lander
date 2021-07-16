import os, gym, time, csv
import pandas as pd
from datetime import date, datetime

env = gym.make('LunarLander-v2')

# Use right, left and down arrows to control the LunarLander
# Total score and duration are saved at a csv file

ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    
    if key == 65363:
        a = 1
    elif key == 65364:
        a = 2
    elif key == 65361:
        a = 3
    else:
        a = 0

    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    
    if key == 65363:
        a = 1
    elif key == 65364:
        a = 2
    elif key == 65361:
        a = 3
    else:
        a = 0
    
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause, rewards_total, duration, rounds
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    start = datetime.now()

    for _ in range(200):
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
    
        #if r != 0:
            #print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    rewards_total += total_reward
    duration += (datetime.now() - start).seconds


def write_to_file():
    time = datetime.now()
    row = [time.strftime("%d-%m-%Y, %H:%M:%S"), rounds, rewards_total, duration]
    if os.path.exists('game_data.csv'):
        with open('game_data.csv', 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)
    else:
        with open('game_data.csv', 'a+') as f:
            write = csv.writer(f)
            write.writerow(['Date', 'Rounds played','Total reward', 'Duration (s)'])
            write.writerow(row)

if __name__ == '__main__':

    rewards_total = 0
    duration = 0
    rounds = 0

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    
    while 1:
        window_still_open = rollout(env)
        if window_still_open==False:
            write_to_file()
            break
        rounds += 1
    
    df = pd.read_csv('game_data.csv')
    print(df.head())