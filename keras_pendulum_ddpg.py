# Reference
# https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://keras.io/examples/rl/ddpg_pendulum/
# https://www.kaggle.com/code/thebratattack/deep-reinforcement-learning-intuition

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import time as tm

import copy
import json 
import datetime 
import pandas as pd 
from tqdm import tqdm 
import os 
import math

random.seed(2212)
np.random.seed(2212)
tf.random.set_seed(2212) 

def save_frames_as_gif(frames, path): 
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path, writer='imagemagick', fps=60) 


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
The `Buffer` class implements Experience Replay.

---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---


**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for a, b in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]



if __name__ == '__main__':

    args = {}

    if not os.path.exists('log'):
        os.makedirs('log')

    if not os.path.exists('model'):
        os.makedirs('model')

    problem = 'Pendulum-v1' # type=str 
    args['env'] = copy.deepcopy(problem)
    env = gym.make(problem, render_mode="rgb_array")
    # device = 'CPU:0'  # type=str 
    #  args['device'] = copy.deepcopy(device)

    reward_txt_filename = 'log/Avg_Episodic_Reward.txt' # type=str 
    args['reward_txt_filename'] = copy.deepcopy(reward_txt_filename)

    cosim_log_filename = 'log/Cosim_Log.txt' # type=str 
    args['cosim_log_filename'] = copy.deepcopy(cosim_log_filename)

    actor_weight_filename = 'model/pendulum_actor.h5'  # type=str 
    args['actor_weight_filename'] = copy.deepcopy(actor_weight_filename)

    critic_weight_filename = 'model/pendulum_critic.h5'  # type=str 
    args['critic_weight_filename'] = copy.deepcopy(critic_weight_filename)

    target_actor_weight_filename = 'model/pendulum_target_actor.h5'  # type=str 
    args['target_actor_weight_filename'] = copy.deepcopy(target_actor_weight_filename)

    target_critic_weight_filename = 'model/pendulum_target_critic.h5'  # type=str 
    args['target_critic_weight_filename'] = copy.deepcopy(target_critic_weight_filename)
      
    args_json_filename = 'log/train_args.json' # type=str 
    args['args_json_filename'] = copy.deepcopy(args_json_filename)

    gif_filename = 'plot/gym_animation.gif' # type=str 
    args['gif_filename'] = copy.deepcopy(gif_filename)

    episode_plot_png_filename = 'plot/Avg_Episodic_Reward.png' # type=str 
    args['episode_plot_png_filename'] = copy.deepcopy(episode_plot_png_filename)

    step_plot_png_filename = 'plot/Avg_Stepic_Reward.png' # type=str 
    args['step_plot_png_filename'] = copy.deepcopy(step_plot_png_filename)

    png_title = 'Average Reward of Pendulum' #  type=str 
    args['png_title'] = copy.deepcopy(png_title)

    csv_filename = 'log/pendulum_log.csv' # type=str 
    args['csv_filename'] = copy.deepcopy(csv_filename)

    simulation_time = 0.45 # type=int  
    args['simulation_time'] = copy.deepcopy(simulation_time)
    
    episode = 100 # type=int  
    args['episode'] = copy.deepcopy(episode)

    save_model_episode = 1 # type=int  
    args['save_model_episode'] = copy.deepcopy(save_model_episode)

    memory_capacity = 500000 # type=int, help='replay memory size' 
    args['memory_capacity'] = copy.deepcopy(memory_capacity)

    batch_size = 64 # type=int, help='minibatch size' 
    args['batch_size'] = copy.deepcopy(batch_size)

    num_states = env.observation_space.shape[0] # type=int, help='the number of state' 
    args['num_states'] = copy.deepcopy(num_states)

    num_actions = env.action_space.shape[0] # type=int, help='the number of action' 
    args['num_actions'] = copy.deepcopy(num_actions)

    upper_bound = env.action_space.high[0] # type=float, help='upper bound' 
    args['upper_bound'] = [upper_bound.tolist()]

    lower_bound = env.action_space.low[0] # type=float, help='lower bound' 
    args['lower_bound'] = [lower_bound.tolist()]

    std_dev = 0.2 # type=float, help='std dev' 
    args['std_dev'] = copy.deepcopy(std_dev)

    actor_lr = 0.001 # type=float, help='actor learning rate' 
    args['actor_lr'] = copy.deepcopy(actor_lr)

    critic_lr = 0.002 # type=float, help='critic learning rate' 
    args['critic_lr'] = copy.deepcopy(critic_lr)

    gamma = 0.99 # Discount factor for future rewards # type=float, help='discounted factor' 
    args['gamma'] = copy.deepcopy(gamma)

    tau = 0.005 # Used to update target networks # type=float, help='moving average for target network' 
    args['tau'] = copy.deepcopy(tau)
    # args = pd.DataFrame(args)  
    with open(args_json_filename,'w') as f: 
        json.dump(args, f, ensure_ascii=False, indent=4) 
    args = pd.DataFrame.from_dict(args, orient='index') 
    args.to_csv(args_json_filename.replace('json', 'csv'))  

    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    buffer = Buffer(50000, 64)

    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    log = open(reward_txt_filename, 'w', encoding='utf-8') 
    log.write('start simulation.\n')
    log.close()
    # DDPG 강화 학습 로그 텍스트 파일 생성
    log = open(cosim_log_filename, 'w', encoding='utf-8') 
    log.write('Start simulation.\n')
    log.close()

    action = np.zeros(num_actions)

    for ep in range(episode): 
        print("Episode num = %d" % ep)

        prev_state = env.reset()
        episodic_reward = 0 
        num_step = 0 

        action_list = [] 
        state_list = [] 
        time_list = [] 
        reward_list = []        
        one_reward_list = [] 

        prev_state = np.zeros(num_states)  # RefPoint1 Initial Position(x,y,z)
        action = np.zeros(num_actions)

        log = open(cosim_log_filename, 'a', encoding='utf-8')  
        log.write("======================================================================\n\n")
        log.write("Episode %d\n" % ep)
        log.close()

        ############################################################################################
        start = tm.time()     

        frames = []
        img = env.render()     

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()
            num_step += 1 
              
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

            log = open(cosim_log_filename, 'a', encoding='utf-8')  
            state_str = ', '.join('%.3f' % s for s in state)
            log.write("current_state  = ( %s )\n" % state_str)
            state_list.append([round(s, 3) for s in state])  
            action_str = ', '.join('%.3f' % a for a in action)
            log.write("current_action = ( %s )\n" % action_str)
            action_list.append([round(a, 3) for a in action])  
            log.write("reward         =   %.3f\n" % reward)
            reward_list.append(round(reward, 3)) 
            log.close()

            img = env.render()

        log = open(cosim_log_filename, 'a', encoding='utf-8')  
        log.write("======================================================================\n\n")
        log.close()

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes 
        avg_reward = np.mean(ep_reward_list[-40:]) 
        # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        env.close()

        end = tm.time() 
        episode_sec = (end - start) 
        step_sec = episode_sec / num_step 
        episode_time_list = str(datetime.timedelta(seconds=episode_sec)).split(".") 
        step_time_list = str(datetime.timedelta(seconds=step_sec)).split(".") 
        episode_time = episode_time_list[0] 
        step_time = step_time_list[0] 

        tm.sleep(0.5)

        log = open(reward_txt_filename, 'a', encoding='utf-8') 
        log.write("Episode * {} * Total Step * {} * Avg Reward {} * Episode Time * {} * Avg Step Time * {} * \n".format(ep, num_step, avg_reward, episode_time, step_time))
        log.close()

        print("Episode * {} * Total Step * {} * Avg Reward {} * Episode Time * {} * Avg Step Time * {} * \n".format(ep, num_step, avg_reward, episode_time, step_time))
       
        plt.figure(figsize=(10, 5))  
        plt.plot(ep_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Episodic Reward")
        plt.savefig(episode_plot_png_filename.replace('.png', '_' + str(ep + 1) + '.png'), facecolor='#eeeeee', edgecolor='black', format='png', bbox_inches='tight')  
        
        plt.figure(figsize=(10, 5))  
        plt.plot(reward_list)
        plt.xlabel("Step")
        plt.ylabel("Stepic Reward")
        plt.savefig(step_plot_png_filename.replace('.png', '_' + str(ep + 1) + '.png'), facecolor='#eeeeee', edgecolor='black', format='png', bbox_inches='tight')  
        
        log_df = pd.DataFrame({"State": state_list,  
                               "Action": action_list,  
                               "Reward": reward_list,  
                               "Time": time_list}) 
        log_df.to_csv(csv_filename.replace('.csv', '_' + str(ep + 1) + '.csv'), index=False) 

        reward_df = pd.DataFrame({"Reward": reward_list}) 
        reward_df.to_csv(csv_filename.replace("_log", "_reward").replace('.csv', '_' + str(ep + 1) + '.csv'), index=False) 

        save_frames_as_gif(frames, gif_filename.replace('.gif', '_' + str(ep + 1) + '.gif')) 

        if ep % save_model_episode == 0:
            actor_model_file = actor_weight_filename.replace('.h5', '_' + str(ep + 1) + '.h5')
            critic_model_file = critic_weight_filename.replace('.h5', '_' + str(ep + 1) + '.h5')
            target_actor_model_file = target_actor_weight_filename.replace('.h5', '_' + str(ep + 1) + '.h5')
            target_critic_model_file = target_critic_weight_filename.replace('.h5', '_' + str(ep + 1) + '.h5')

            actor_model.save_weights(actor_model_file)
            critic_model.save_weights(critic_model_file)

            target_actor.save_weights(target_actor_weight_filename)
            target_critic.save_weights(target_critic_weight_filename)


    # Plotting graph: Episodes versus Avg. Rewards
    plt.figure(figsize=(10, 5))  
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.savefig(episode_plot_png_filename, facecolor='#eeeeee', edgecolor='black', format='png', bbox_inches='tight') #revised

    # Save the weights
    # actor_model.save_weights(actor_model_file)
    # critic_model.save_weights(critic_model_file)
    # target_actor.save_weights(target_actor_weight_filename)
    # target_critic.save_weights(target_critic_weight_filename)