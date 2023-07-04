import sys
import json
import os
import pandas as pd
import numpy as np
from os.path import isfile, join
from math import isclose
import warnings

from discrete_MDP_helper_functions import find_state_abstraction, state_action_reward, policy_sample_action, state_transition

CONFIG_FOLDER = f'configs/{sys.argv[1]}'

get_path = lambda filename : f'{CONFIG_FOLDER}/{filename}'

#LOAD STATE AND ACTION NUMBERS
state_numbers = None
with open(get_path('state-numbers.json'), 'r') as f:
    state_numbers = json.load(f)
action_numbers = None
with open(get_path('action-numbers.json'), 'r') as f:
    action_numbers = json.load(f)

# LOAD PROPERTIES OF OVERALL MDP
state_action_map = None
with open(get_path('S-A-config.json'), 'r') as f:
    state_action_map = json.load(f)

df_P = pd.read_csv( get_path('P.csv') )
P = np.zeros(shape=(len(state_numbers.keys()), len(action_numbers.keys()), len(state_numbers.keys())))
for rnum in range(len(df_P)):
    row = df_P.loc[rnum]
    P[state_numbers[row['state']], action_numbers[row['action']], state_numbers[row['next_state']]] = float(row['probability'])
df_R = pd.read_csv( get_path('R.csv') )
R = np.zeros(shape=(len(state_numbers.keys()), len(action_numbers.keys()), len(state_numbers.keys())))
for rnum in range(len(df_R)):
    row = df_R.loc[rnum]
    R[state_numbers[row['state']], action_numbers[row['action']], state_numbers[row['next_state']]] = float(row['reward'])

# LOAD BEHAVIOUR AND EVALUATION POLICIES
df_behaviour_policy = pd.read_csv( get_path('behaviour-policy.csv') )
pi_b = np.zeros(shape=(len(state_numbers.keys()), len(action_numbers.keys())))
for rnum in range(len(df_behaviour_policy)):
    row = df_behaviour_policy.loc[rnum]
    pi_b[state_numbers[row['state']], action_numbers[row['action']]] = float(row['probability'])
df_evaluation_policy = pd.read_csv( get_path('evaluation-policy.csv') )
pi_e = np.zeros(shape=(len(state_numbers.keys()), len(action_numbers.keys())))
for rnum in range(len(df_evaluation_policy)):
    row = df_evaluation_policy.loc[rnum]
    pi_e[state_numbers[row['state']], action_numbers[row['action']]] = float(row['probability'])

# LOAD FACTORISATION INFORMATION
FACTORISATION_FOLDER = f'{CONFIG_FOLDER}/factorisation'
action_spaces = [f for f in os.listdir(FACTORISATION_FOLDER) if not isfile(join(FACTORISATION_FOLDER, f))]

action_space_mapping = None
with open(f'{FACTORISATION_FOLDER}/action-mapping.json', 'r') as f:
    action_space_mapping = json.load(f)

state_abstractions = None
with open(f'{FACTORISATION_FOLDER}/state-abstractions.json', 'r') as f:
    state_abstractions = json.load(f)

factored_Ps = []
factored_Rs = []
factored_behaviour_policies = []
factored_evaluation_policies = []
factored_state_action_maps = []
factored_state_abstractions = []
factored_state_numbers = []
factored_action_numbers = []

for action_space in action_spaces:
    ACTION_SPACE_FOLDER = f'{FACTORISATION_FOLDER}/{action_space}'
    get_path = lambda filename : f'{ACTION_SPACE_FOLDER}/{filename}'

    # LOAD PROPERTIES OF SUB MDP
    f_state_numbers = None
    with open(get_path('state-numbers.json'), 'r') as f:
        f_state_numbers = json.load(f)
        factored_state_numbers.append(f_state_numbers)
        
    f_action_numbers = None
    with open(get_path('action-numbers.json'), 'r') as f:
        f_action_numbers = json.load(f)
        factored_action_numbers.append(f_action_numbers)
        
    with open(get_path('S-A-config.json'), 'r') as f:
        factored_state_action_maps.append( json.load(f) )

    factored_state_abstractions.append(state_abstractions[action_space])

    # LOAD TRANSITION PROBABILITIES
    f_df_P = pd.read_csv( get_path('P.csv') )
    f_P = np.zeros( shape = (len(f_state_numbers.keys()), len(f_action_numbers.keys()), len(f_state_numbers.keys()) ) )
    for rnum in range(len(f_df_P)):
        row = f_df_P.loc[rnum]
        f_P[f_state_numbers[row['state']], f_action_numbers[row['action']], f_state_numbers[row['next_state']]] = float(row['probability'])
    factored_Ps.append(f_P)

    # LOAD REWARDS
    f_df_R = pd.read_csv( get_path('R.csv') )
    f_R = np.zeros( shape = ( len(f_state_numbers.keys()), len(f_action_numbers.keys()), len(f_state_numbers.keys()) ) )
    for rnum in range(len(f_df_R)):
        row = f_df_R.loc[rnum]
        f_R[f_state_numbers[row['state']], f_action_numbers[row['action']], f_state_numbers[row['next_state']]] = float(row['reward'])
    factored_Rs.append(f_R)

    # LOAD BEHAVIOUR AND EVALUATION POLICIES
    f_pi_b_df = pd.read_csv( get_path('behaviour-policy.csv') )
    f_pi_b = np.zeros(shape=( len(f_state_numbers.keys()), len(f_action_numbers.keys()) ))
    for rnum in range(len(f_pi_b_df)):
        row = f_pi_b_df.loc[rnum]
        f_pi_b[f_state_numbers[row['state']], f_action_numbers[row['action']]] = float(row['probability'])
    factored_behaviour_policies.append(f_pi_b)

    f_pi_e_df = pd.read_csv( get_path('evaluation-policy.csv') )
    f_pi_e = np.zeros(shape = ( len(f_state_numbers.keys()), len(f_action_numbers.keys()) ))
    for rnum in range(len(f_pi_e_df)):
        row = f_pi_e_df.loc[rnum]
        f_pi_e[f_state_numbers[row['state']], f_action_numbers[row['action']]] = float(row['probability'])
    factored_evaluation_policies.append(f_pi_e)



#CHECK THAT ALL THE MDPS ARE WELL FORMED

# Check that probabilities in P and policies sum to 1
def check_MDP_probabilities(state_action_map, pi_b, pi_e, P, state_numbers, action_numbers):
    states = state_action_map.keys()
    for state in states:
        state_no = state_numbers[state]
        behaviour_probability_sum = 0.0
        evaluation_probability_sum = 0.0
        for action in state_action_map[state]:
            action_no = action_numbers[action]
            #Handle policy probabilities
            behaviour_probability_sum += pi_b[state_no, action_no]
            evaluation_probability_sum += pi_e[state_no, action_no]
            #Handle transition probabilities
            P_probability_sum = np.sum(P[state_no, action_no, :])
            #Raise an error if the probabilities for all s' in (s,a,s') do not sum to 1
            if ( not isclose(P_probability_sum, 1.0) ) and len(state_action_map[state]) > 0:
                raise ValueError(f'The probability of subsequent states from state:{state}, action:{action} does not sum to 1. The sum is {P_probability_sum}.')
        #Raise an error if the probabilities for all a in (s,a) do not sum to 1
        if (not isclose(behaviour_probability_sum, 1.0) ) and len(state_action_map[state]) > 0:
            raise ValueError(f'In behaviour policy, the probability of subsequent actions from state:{state} does not sum to 1. The sum is {behaviour_probability_sum}.')
        if (not isclose(evaluation_probability_sum, 1.0) ) and len(state_action_map[state]) > 0:
            raise ValueError(f'In evaluation policy, the probability of subsequent actions from state:{state} does not sum to 1. The sum is {evaluation_probability_sum}.')

  
check_MDP_probabilities(state_action_map, pi_b, pi_e, P, state_numbers, action_numbers)
for ind in range(len(action_spaces)):
    check_MDP_probabilities(factored_state_action_maps[ind],
                            factored_behaviour_policies[ind],
                            factored_evaluation_policies[ind],
                            factored_Ps[ind],
                            factored_state_numbers[ind],
                            factored_action_numbers[ind])
 
#CHECK WHETHER THEOREM 1 IS SATISFIED AND IF NOT, EXPLAIN WHY

theorem_1_satisfied = True

#Check transition probabilities
#   Iterate over (s,a,s')
for state in state_action_map.keys():
    state_no = state_numbers[state]
    for action in state_action_map[state]:
        action_no = action_numbers[action]
        for next_state in state_action_map.keys():
            next_state_no = state_numbers[next_state]
            abstraction_prob_prod = 1.0
            next_state_set = None
            #Here we assume that the action is specified in terms of sub-actions in a string separated by commas
            for ind in range(len(action_spaces)):
                action_space = action_spaces[ind]
                sub_action = action_space_mapping[action][action_space]
                sub_action_no = factored_action_numbers[ind][sub_action]
                state_abstraction, _ = find_state_abstraction(factored_state_abstractions[ind], state)
                state_abstraction_no = factored_state_numbers[ind][state_abstraction]
                if sub_action not in factored_state_action_maps[ind][state_abstraction]:
                    continue
                next_state_abstraction, next_state_list = find_state_abstraction(factored_state_abstractions[ind], next_state)
                next_state_abstraction_no = factored_state_numbers[ind][next_state_abstraction]
                if next_state_set is None:
                    next_state_set = set(next_state_list)
                next_state_set = next_state_set.intersection(set(next_state_list))
                abstraction_prob_prod *= factored_Ps[ind][state_abstraction_no, sub_action_no, next_state_abstraction_no]
            #calculate LHS of expression for transition probabilities in theorem 1.
            LHS_prob_sum = 0.0
            #Iterate over states obtained by transforming to and from the abstract state space
            for next_state_proxy in list(next_state_set):
                next_state_proxy_no = state_numbers[next_state_proxy]
                LHS_prob_sum += P[state_no, action_no, next_state_proxy_no]
            #Give warning if condition not satisfied
            if not isclose(LHS_prob_sum, abstraction_prob_prod):
                warnings.warn(f'Theorem 1 is not satisfied in transition probabilities of ({state},{action},{next_state}). LHS: {LHS_prob_sum}, RHS: {abstraction_prob_prod}')  

 
#Check rewards
for state in state_action_map.keys():
    state_no = state_numbers[state]
    for action in state_action_map[state]:
        action_no = action_numbers[action]
        main_MDP_reward = state_action_reward(R, P, state_action_map, state_no, action_no)
        factored_reward_sum = 0.0
        #Here we assume that the action is specified in terms of sub-actions in a string separated by commas
        for ind in range(len(action_spaces)):
            action_space = action_spaces[ind]
            sub_action = action_space_mapping[action][action_space]
            sub_action_no = factored_action_numbers[ind][sub_action]
            state_abstraction, _ = find_state_abstraction(factored_state_abstractions[ind], state)
            state_abstraction_no = factored_state_numbers[ind][state_abstraction]
            factored_reward_sum += state_action_reward(factored_Rs[ind], factored_Ps[ind], factored_state_action_maps[ind], state_abstraction_no, sub_action_no)
        if not isclose(main_MDP_reward, factored_reward_sum):
            warnings.warn(f'Theorem 1 is not satisfied in rewards of ({state},{action},{next_state}). Main MDP reward: {main_MDP_reward}, factored reward sum: {factored_reward_sum}')


#Check policy probabilities
for state in state_action_map.keys():
    state_no = state_numbers[state]
    for action in state_action_map[state]:
        action_no = action_numbers[action]
        behaviour_policy_prob = pi_b[state_no, action_no]
        evaluation_policy_prob = pi_e[state_no, action_no]
        factored_behaviour_policy_prod = 1.0
        factored_evaluation_policy_prod = 1.0
        #Here we assume that the action is specified in terms of sub-actions in a string separated by commas
        for ind in range(len(action_spaces)):
            action_space = action_spaces[ind]
            sub_action = action_space_mapping[action][action_space]
            sub_action_no = factored_action_numbers[ind][sub_action]
            state_abstraction, _ = find_state_abstraction(factored_state_abstractions[ind], state)
            state_abstraction_no = factored_state_numbers[ind][state_abstraction]
            if sub_action not in factored_state_action_maps[ind][state_abstraction]:
                continue
            factored_behaviour_policy_prod *= factored_behaviour_policies[ind][state_abstraction_no, sub_action_no]
            factored_evaluation_policy_prod *= factored_evaluation_policies[ind][state_abstraction_no, sub_action_no]
        if not isclose(factored_behaviour_policy_prod, behaviour_policy_prob):
            warnings.warn(f'Theorem 1 is not satisfied in behaviour policy at ({state},{action}). LHS: {behaviour_policy_prob}, RHS: {factored_behaviour_policy_prod}')
        if not isclose(factored_evaluation_policy_prod, evaluation_policy_prob):
            warnings.warn(f'Theorem 1 is not satisfied in evaluation policy of ({state},{action}). LHS: {evaluation_policy_prob}, RHS: {factored_evaluation_policy_prod}')                                   
            
            





