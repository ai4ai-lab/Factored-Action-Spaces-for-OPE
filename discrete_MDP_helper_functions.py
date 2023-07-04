from random import choices
import numpy as np


#Find the state abstraction for a factored action space corresponding to a state in the main MDP
def find_state_abstraction(state_abstraction_map, state):
    keys = list(state_abstraction_map.keys())
    values = list(state_abstraction_map.values())
    for lst_id in range(len(values)):
        if state in values[lst_id]:
            return keys[lst_id], values[lst_id]        

#Return the expected reward given a state number and action number
def state_action_reward(R, P, state_action_map, state_no, action_no):
    probabilities = P[state_no, action_no, :]
    rewards = R[state_no, action_no, :]
    return np.average(rewards, weights=probabilities)


#Sample action from policy given a state number
def policy_sample_action(policy, state_no):
    probabilities = policy[state_no, :]
    actions = [i for i in range(len(probabilities))]
    return choices(actions, probabilities)[0]


#Sample the next state given a state and action
def state_transition(P, state_no, action_no):
    probabilities = P[state_no, action_no, :]
    next_states = [i for i in range(len(probabilities))]
    return choices(next_states, probabilities)[0]
