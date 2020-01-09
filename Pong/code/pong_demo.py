""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import time

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
	model = pickle.load(open('save.p', 'rb'))
else:
	model = {}
	model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
	model['W2'] = np.random.randn(H) / np.sqrt(H)
	
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
	return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	I = I[35:195] # crop
	I = I[::2,::2,0] # downsample by factor of 2
	I[I == 144] = 0 # erase background (background type 1)
	I[I == 109] = 0 # erase background (background type 2)
	I[I != 0] = 1 # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel()

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

def policy_forward(x):
	h = np.dot(model['W1'], x)
	h[h<0] = 0 # ReLU nonlinearity
	logp = np.dot(model['W2'], h)
	p = sigmoid(logp)
	return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
	""" backward pass. (eph is array of intermediate hidden states) """
	dW2 = np.dot(eph.T, epdlogp).ravel()
	dh = np.outer(epdlogp, model['W2'])
	dh[eph <= 0] = 0 # backpro prelu
	dW1 = np.dot(dh.T, epx)
	return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
	if render:
		env.render()
		time.sleep(0.01)

	# preprocess the observation, set input to network to be difference image
	cur_x = prepro(observation)
	x = cur_x - prev_x if prev_x is not None else np.zeros(D)
	prev_x = cur_x

	# forward the policy network and sample an action from the returned probability
	aprob, h = policy_forward(x)
	action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

	# record various intermediates (needed later for backprop)
	xs.append(x) # observation
	hs.append(h) # hidden state
	y = 1 if action == 2 else 0 # a "fake label"
	dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

	# step the environment and get new measurements
	observation, reward, done, info = env.step(action)
	reward_sum += reward

	drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

	if done: # an episode finished
		break

print(('ep %d: game finished, reward: %f' % (episode_number, reward_sum)) + ('' if reward == -1 else ' !!!!!!!!'))

