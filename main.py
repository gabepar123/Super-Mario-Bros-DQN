import imp
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym_super_mario_bros
import cv2
from mario_agent import mario_agent
from replay_buffer import replay_buffer

STATE_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
NEXT_STATE_INDEX = 3
DONE_INDEX = 4

SCREEN_SIZE = 84 #size of screen after processing
EPISODES = 2000
STEPS = 500
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


def process_state(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) #conver to grayscale
    resized_state = cv2.resize(gray, (SCREEN_SIZE, SCREEN_SIZE)) #resize 
    flattened_state = resized_state.flatten()
    #return resized_state
    return flattened_state


def train():
    done = False
    agent = mario_agent()
    replay_memory = replay_buffer()


    for episode in range(EPISODES):
        
        preprocessed_state = env.reset()
        state = process_state(preprocessed_state)
        env.render()

        while not done:
            action = agent.get_action(state)
            preprocessed_state, reward, done, _ = env.step(action)
            env.render()
            next_state = process_state(preprocessed_state)

            replay_memory.push(state, action, reward, next_state, done)
            
            replay_batch = replay_memory.random_sample()
            for replay in replay_batch:
                if replay[DONE_INDEX]:
                    y = replay[REWARD_INDEX]
                else:
                    y = replay[REWARD_INDEX] + 0.95 * agent.get_target_max_q_val(replay[NEXT_STATE_INDEX])
                
                #preform gradient here
                #agent.compute_loss(replay[STATE_INDEX], y - agent.get_q_vals(replay[STATE_INDEX]))
            state = next_state

        env.close()

train()