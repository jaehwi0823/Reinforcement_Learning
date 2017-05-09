import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

# Hyper parameters
learning_rate = 0.25
discount = 0.9
num_episodes = 300
batch_size = 10

# Tensorflow input
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 2])
W = tf.Variable(tf.random_normal([4, 2], mean=0, stddev=0.01), trainable=True)

# Inference Model
Yhat = tf.matmul(X, W)

# Loss Function (Objective Function)
loss = tf.reduce_sum(tf.square(Y-Yhat))

# Optimizer
Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 모형 초기화
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
epoch = 1
e = 0.5
loss_list = []
batch_cnt = 0
batch_matrix_obs = []
batch_matrix_lbl = []

# 게임반복 횟수 설정
while epoch < num_episodes:
    # 매 게임마다 환경초기화
    done = False
    observation = np.reshape(env.reset(), [1, 4])
    reward_sum = 0

    while not done:
        # display the game(optional)
        # env.render()

        # predict next action-reward, action
        action_reward = sess.run(Yhat, feed_dict={X: observation})
        # print("Action reward = ", action_reward)

        # random walk
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(action_reward)

        # check the result of the prediction
        next_observation, reward, done, _ = env.step(action)
        next_observation_rs = np.reshape(next_observation, [1, 4])
        reward_sum += reward

        # according to the game result
        if done:
            action_reward[0, action] = -10
            print("Reward for ", epoch, "th episode was:", reward_sum)
            # print("=================================================================================================")
        else:
            next_action_reward = sess.run(Yhat, feed_dict={X: next_observation_rs})
            action_reward[0, action] = reward + discount * np.max(next_action_reward)

        # batch control
        if batch_cnt < batch_size:
            batch_matrix_obs.append(observation)
            batch_matrix_lbl.append(action_reward)
            batch_cnt += 1
        else:
            # train the model
            sess.run(Optimizer, feed_dict={X: np.reshape(batch_matrix_obs, [batch_size, 4]),
                                           Y: np.reshape(batch_matrix_lbl, [batch_size, 2])})

            # Loss Graph control
            loss_value = sess.run(loss, feed_dict={ X: np.reshape(batch_matrix_obs, [batch_size, 4]),
                                                    Y: np.reshape(batch_matrix_lbl, [batch_size, 2])})
            loss_list.append(loss_value)

            # batch reset
            batch_matrix_obs = []
            batch_matrix_lbl = []
            batch_cnt = 0

        # move a step forward
        observation = next_observation_rs

    # iteration control
    epoch += 1
    learning_rate *= 0.99
    # print(learning_rate)
    e = 1. / ((epoch // 10) + 1)

plt.plot(range(len(loss_list)), loss_list, color='b', alpha=0.4)
plt.show()
sess.close()
