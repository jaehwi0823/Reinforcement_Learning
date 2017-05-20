import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

# Hyper parameters
learning_rate = 0.2

# Tensorflow input
X = tf.placeholder(tf.float32, shape=[1, 4])
Y = tf.placeholder(tf.float32, shape=[1, 2])
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
discount = 0.5
e = 0.5
loss_list = []

# 게임반복 횟수 설정
while epoch < 50:
    # 매 게임마다 환경초기화
    done = False
    # 4개 값 리턴 --> 일종의 state
    observation = np.reshape(env.reset(), [1, 4])
    reward_sum = 0

    #게임이 끝날때까지
    while not done:
        # display the game(optional)
        env.render()

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
            # print("==============================================================================================================")
        else:
            next_action_reward = sess.run(Yhat, feed_dict={X: next_observation_rs})
            action_reward[0, action] = min(reward + discount * np.max(next_action_reward), 999)

        # train the model and move a step forward
        sess.run(Optimizer, feed_dict={X: observation, Y: action_reward})
        loss_value = sess.run(loss, feed_dict={X: observation, Y: action_reward})
        loss_list.append(loss_value)
        observation = next_observation_rs

    if reward_sum > 10000:  # Good enough. Let's move on
        break

    # iteration control
    epoch += 1
    learning_rate *= 0.999
    e = 1. / ((epoch // 10) + 1)

plt.plot(range(len(loss_list)), loss_list, color='b', alpha=0.4)
plt.show()
