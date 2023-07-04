from load_discrete_MDP import state_action_map, P, R, pi_b, pi_e
from load_discrete_MDP import action_spaces, action_space_mapping, state_abstractions
from load_discrete_MDP import factored_Ps, factored_Rs, factored_behaviour_policies, factored_evaluation_policies, factored_state_action_maps, factored_state_abstractions
from load_discrete_MDP import state_numbers, action_numbers, factored_state_numbers, factored_action_numbers
from load_discrete_MDP import find_state_abstraction, policy_sample_action, state_transition 
import numpy as np
import random
import pickle
import sys

NO_TRAJECTORIES = int(sys.argv[2])
TRAJECTORY_LENGTH = int(sys.argv[3])

inv_state_numbers = {v: k for k, v in state_numbers.items()}
inv_action_numbers = {v: k for k, v in action_numbers.items()} 

#Calculate policy divergence from Voloshin et al.
D = 0
for state in state_action_map.keys():
    state_no = state_numbers[state]
    for action in state_action_map[state]:
        action_no = action_numbers[action]
        difference = pi_e[state_no, action_no]/pi_b[state_no, action_no]
        D = max(D, difference)

print(f'Policy divergence: {D}')

#Randomly choose a non-terminal start state
start_state_no = None
start_state = None
while start_state_no is None or len(state_action_map[start_state]) == 0:
    start_state = random.choice(list(state_action_map.keys()))
    start_state_no = state_numbers[start_state]
#Print out
print(f'Start state: {start_state}, start state no: {start_state_no}')

#GENERATE DATA BOTH FOR EVALUATION AND BEHAVIOUR POLICY
#We generate the non-factored data set as a NumPy array of the different trajectories, each time step of which is represented by [t,s,a,r,s']   
nf_transitions_b = np.zeros( shape=(NO_TRAJECTORIES, TRAJECTORY_LENGTH, 5) )
nf_transitions_e = np.zeros( shape=(NO_TRAJECTORIES, TRAJECTORY_LENGTH, 5) )
#We generate the factored data set as a NumPy array of the different trajectories in each factored action space, each time step represented by [t,z,a,r,z'] 
f_transitions_b = np.zeros( shape=(NO_TRAJECTORIES, TRAJECTORY_LENGTH, len(action_spaces), 5) )
f_transitions_e = np.zeros( shape=(NO_TRAJECTORIES, TRAJECTORY_LENGTH, len(action_spaces), 5) )

for trajectory in range(NO_TRAJECTORIES):
    #Initialise to start state for both policies
    current_state_no_b = start_state_no
    current_state_no_e = start_state_no
    #Sample trajectories from MDP, following policies    
    for time_step in range(TRAJECTORY_LENGTH):
        #----Save initial information for both policies----
        #Behaviour
        nf_transitions_b[trajectory, time_step, 0] = time_step
        nf_transitions_b[trajectory, time_step, 1] = current_state_no_b
        #Evaluation
        nf_transitions_e[trajectory, time_step, 0] = time_step
        nf_transitions_e[trajectory, time_step, 1] = current_state_no_e
        #----Obtain and save action for both policies----
        #Behaviour
        action_no_b = policy_sample_action(pi_b, current_state_no_b)
        nf_transitions_b[trajectory, time_step, 2] = action_no_b
        #Evaluation
        action_no_e = policy_sample_action(pi_e, current_state_no_e)
        nf_transitions_e[trajectory, time_step, 2] = action_no_e
        #----Obtain and save next state----
        #Behaviour
        next_state_no_b = state_transition(P, current_state_no_b, action_no_b)
        nf_transitions_b[trajectory, time_step, 4] = next_state_no_b
        #Evaluation
        next_state_no_e = state_transition(P, current_state_no_e, action_no_e)
        nf_transitions_e[trajectory, time_step, 4] = next_state_no_e
        #----Save reward----
        nf_transitions_b[trajectory, time_step, 3] = R[current_state_no_b, action_no_b, next_state_no_b]
        nf_transitions_e[trajectory, time_step, 3] = R[current_state_no_e, action_no_e, next_state_no_e]
        #----Express everything in terms of factored action spaces----
        #Behaviour
        action_b = inv_action_numbers[action_no_b]
        current_state_b = inv_state_numbers[current_state_no_b]
        next_state_b = inv_state_numbers[next_state_no_b]
        #Evaluation
        action_e = inv_action_numbers[action_no_e]
        current_state_e = inv_state_numbers[current_state_no_e]
        next_state_e = inv_state_numbers[next_state_no_e]
        #-----------------------------------
        for ind in range(len(action_spaces)):
            action_space = action_spaces[ind]
            #----Obtain sub action----
            #Behaviour
            sub_action_b = action_space_mapping[action_b][action_space]
            sub_action_no_b = factored_action_numbers[ind][sub_action_b]
            #Evaluation
            sub_action_e = action_space_mapping[action_e][action_space]
            sub_action_no_e = factored_action_numbers[ind][sub_action_e]
            #----Obtain current state abstraction----
            #Behaviour
            current_state_abstraction_b, _ = find_state_abstraction(factored_state_abstractions[ind], current_state_b)
            current_state_abstraction_no_b = factored_state_numbers[ind][current_state_abstraction_b]
            #Evaluation
            current_state_abstraction_e, _ = find_state_abstraction(factored_state_abstractions[ind], current_state_e)
            current_state_abstraction_no_e = factored_state_numbers[ind][current_state_abstraction_e]
            #----Obtain next state abstraction----
            #Behaviour
            next_state_abstraction_b, _ = find_state_abstraction(factored_state_abstractions[ind], next_state_b)
            next_state_abstraction_no_b = factored_state_numbers[ind][next_state_abstraction_b]
            #Evaluation
            next_state_abstraction_e, _ = find_state_abstraction(factored_state_abstractions[ind], next_state_e)
            next_state_abstraction_no_e = factored_state_numbers[ind][next_state_abstraction_e]
            #----Obtain sub reward----
            #Behaviour
            sub_reward_b = factored_Rs[ind][current_state_abstraction_no_b, sub_action_no_b, next_state_abstraction_no_b]
            #Evaluation
            sub_reward_e = factored_Rs[ind][current_state_abstraction_no_e, sub_action_no_e, next_state_abstraction_no_e]
            #----Store----
            #Behaviour
            f_transitions_b[trajectory, time_step, ind, :] = [time_step,
                                                              current_state_abstraction_no_b,
                                                              sub_action_no_b,
                                                              sub_reward_b,
                                                              next_state_abstraction_no_b]
            #Evaluation
            f_transitions_e[trajectory, time_step, ind, :] = [time_step,
                                                              current_state_abstraction_no_e,
                                                              sub_action_no_e,
                                                              sub_reward_e,
                                                              next_state_abstraction_no_e]
        #----Update state----
        current_state_no_b = next_state_no_b
        current_state_no_e = next_state_no_e


#Save non factored transitions as npy file
version_number = 1
shorter_D = round(D, 2)
DATASET_FILENAME = lambda v : f'datasets/{sys.argv[1]}-{start_state_no}-{NO_TRAJECTORIES}-{TRAJECTORY_LENGTH}-{shorter_D}-{v}.npy'
unused_version = False
while not unused_version:
    try:
        f = open(DATASET_FILENAME(version_number), "rb")
        f.close()
        version_number += 1
    except:
        unused_version = True

with open(DATASET_FILENAME(version_number), "wb") as fp:
    np.save(fp, nf_transitions_b)
    np.save(fp, nf_transitions_e)
    np.save(fp, f_transitions_b)
    np.save(fp, f_transitions_e)

