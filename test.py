import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

print(env.action_space.sample())
print(np.random.rand(1))


# # 게임 이해하기
# # observation = np.reshape(env.reset(), [1, 4])
# # Input 확인
# print(env.reset())
# # Output 확인
# action = env.action_space.sample()
# print(action)
# # 결과확인하기
# next_observation, reward, done, _ = env.step(2)
# print(next_observation)
# print(reward)
# print(done)

# Observation: shape이 [4] 인 vector
# Action (Command): shape이 [2] 인 Vector
# 4x2 NN: y=XW

# Hyperparameter
# learning_rate = 0.15
# discount = 0.9
# itr = 100
# e = 0.5
#
# # Input Define
# X = tf.placeholder(dtype=tf.float32, shape=[1, 4])
# Y = tf.placeholder(dtype=tf.float32, shape=[1, 2])
# W = tf.Variable(tf.truncated_normal(shape=[4, 2]))
#
# # Inference Model
# Yhat = tf.matmul(X, W)
#
# # Cost function
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y - Yhat), reduction_indices=[1]))
#
# # Optimizer
# trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
#
# with tf.Session() as sess:
#     # TF변수 초기화
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     for epoch in range(itr):
#         # 게임 초기화: [4]
#         observation = np.reshape(env.reset(), [1, 4])
#         reward_sum = 0
#         done = 0
#
#         while not done:
#             # (optional) game 보여주기
#             env.render()
#
#             # 학습된 모형을 활용해서 액션 도출
#             action_reward = sess.run(Yhat, feed_dict={X: observation})
#             if np.random.rand(1) < e:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(action_reward)
#
#             # 도출된 액션으로 학습
#             next_observation, reward, done, _ = env.step(action)
#             next_observation_rs = np.reshape(next_observation, [1, 4])
#             reward_sum += reward
#
#             # according to the game result
#             if done:
#                 action_reward[0, action] = -10
#                 print("Reward for ", epoch, "th episode was:", reward_sum)
#             else:
#                 next_action_reward = sess.run(Yhat, feed_dict={X: next_observation_rs})
#                 action_reward[0, action] = reward + discount * np.max(next_action_reward)
#
#             # train the model and move a step forward
#             sess.run(trainer, feed_dict={X: observation, Y: action_reward})
#             observation = next_observation_rs
#         e = 1. / ((epoch // 10) + 1)
#         learning_rate *= 0.999