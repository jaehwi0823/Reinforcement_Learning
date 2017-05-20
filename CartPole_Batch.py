import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

# Hyper parameters
learning_rate = 0.1
discount = 0.9
num_episodes = 1000
e = 0.5
reward_list = []

# Batch control
batch_size = 10
batch_cnt = 0
batch_matrix_obs = []
batch_matrix_lbl = []

# Tensorflow input
with tf.name_scope("Inputs"):
    X = tf.placeholder(tf.float32, shape=[None, 4])
    Y = tf.placeholder(tf.float32, shape=[None, 2])
    W = tf.Variable(tf.random_normal([4, 2], mean=0, stddev=0.01), trainable=True)
    global_step = tf.Variable(0, name='global_step', trainable=False)

# Inference Model
with tf.name_scope("Inference_Model"):
    Yhat = tf.matmul(X, W)

# Loss Function (Objective Function)
with tf.name_scope("Ioss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y-Yhat), reduction_indices=[1]))
    #loss = tf.contrib.losses.softmax_cross_entropy()

# Optimizer
with tf.name_scope("Optimizer"):
    Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    #Optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    # Variables Initialization
    init = tf.global_variables_initializer()
    sess.run(init)

    # Visualization
    writer = tf.summary.FileWriter(logdir="C:/Users/Jaehwi/Documents/4.Lecture/cs20si", graph=sess.graph)
    tf.summary.scalar("loss", loss)

    for i in range(num_episodes):
        # 매 게임마다 환경초기화
        done = False
        observation = np.reshape(env.reset(), [1, 4])
        reward_sum = 0
        while not done:
            # display the game(optional)
            # env.render()

            # predict next action-reward, action
            action_reward = sess.run(Yhat, feed_dict={X: observation})
            print("Action reward = ", action_reward)

            # random walk
            if np.random.rand(1) < e:
                if np.random.rand(1) < 0.5:
                    action = 0
                else:
                    action = 1
            else:
                action = np.argmax(action_reward)

            # check the result of the prediction
            next_observation, reward, done, _ = env.step(action)
            next_observation_rs = np.reshape(next_observation, [1, 4])
            reward_sum += reward

            # according to the game result
            if done:
                action_reward[0, action] = -100
                reward_list.append(reward_sum)
                #print("Reward for ", i, "th episode was:", reward_sum)
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

                # batch reset
                batch_matrix_obs = []
                batch_matrix_lbl = []
                batch_cnt = 0

        # move a step forward
        observation = next_observation_rs
        learning_rate *= 0.999
        e = 1. / ((epoch // 10) + 1)

plt.plot(range(len(reward_list)), reward_list, color='b', alpha=0.4)
plt.show()
writer.close()
