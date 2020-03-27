import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import Wrapped_Pong as game
import random
import numpy as np

GAMMA = 0.99 # Discount Factor used in calculation of Discounted Future Reward.
ACTIONS = 3 # Number of valid actions (UP, DOWN, NO).
OBSERVE = 50000. # Replay start size: Uniform Random Policy.
EXPLORE = 500000. # The number of frames over which the initial value of EPSILON is linearly annealed.
FINAL_EPSILON = 0.1 # # The final value of EPSILON in EPSILON-greedy Exploration.
INITIAL_EPSILON = 1.0 # The initial value of EPSILON in EPSILON-greedy Exploration.
REPLAY_MEMORY = 500000 # SGD updates are sampled from this number of most recent frames.
BATCH = 100 # The number of training cases over which each stochastic gradient descent update is computed.
K = 1 # only select an action every Kth frame, repeat prev for others

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    
    # Network Weights
    # First Conv Layer
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    
    # Second Conv Layer
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    
    # Third Conv Layer
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    # Fully Connected Layer 1
    W_fc1 = weight_variable([256, 256])
    b_fc1 = bias_variable([256])

    # Fully Connected Layer 2 (Output layer)
    W_fc2 = weight_variable([256, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # Network Architecture
    # Input to Q-Network
    s = tf.placeholder("float", [None, 80, 80, 4])

    # Convolutional Layer 1
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolutional Layer 2
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Convolutional Layer 3
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # Flattened Layer
    h_pool3_flat = tf.reshape(h_pool3, [-1, 256])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    
    # Define the Cost Function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    
    # Mean Squared Error (MSE)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    
    # Adam Optimizer
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # Open up a Game State to communicate with Emulator
    game_state = game.GameState()

    # Store Experiences in Replay Memory
    D = []

    # Image Preprocessing
    x_t, r_0, terminal = game_state.frame_step([1, 0, 0])
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # Saving and Loading networks.
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully Loaded:", checkpoint.model_checkpoint_path

    epsilon = INITIAL_EPSILON
    t = 0
    
    while "PIGS" != "FLY":
        
        # Epsilon-Greddy Exploration : Choose random action with probability EPSILON.
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            # Choose action with maximum Q-value with probability 1-EPSILON.
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # Scale down Epsilon.
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            
            # Run the selected Action and observe Next State and Reward.
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)

            # Store Experiences in Replay Memory.
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.pop(0)

        # Training Phase.
        if t > OBSERVE:
            # Sample a minibatch to train on: Over which SGD update is computed.
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            
            # Estimate the actionvalue function, by using the Bellman equation.
            for i in range(0, len(minibatch)):
                # if terminal only equals reward (Discounted Future Reward).
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # Stoshastic Gradient Descent Update.
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})

        # Update old values.
        s_t = s_t1
        t += K

        # Save progress every 10000 iterations.
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/pong-dqn', global_step = t)

        # Print Information to Console.
        state = ""
        if t <= OBSERVE:
            state = "OBSERVE"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "EXPLORE"
        else:
            state = "TRAIN"
        print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()