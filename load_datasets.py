import numpy as np

def load_MDP_transitions(MDP_name,
                         start_state_no,
                         no_trajectories,
                         trajectory_length,
                         shorter_D,
                         version_number):
    path = f'datasets/{MDP_name}-{start_state_no}-{no_trajectories}-{trajectory_length}-{shorter_D}-{version_number}.npy'
    nf_transitions_b, nf_transitions_e, f_transitions_b, f_transitions_e = None, None, None, None
    with open(path, "rb") as fp:
        nf_transitions_b = np.load(fp)
        nf_transitions_e = np.load(fp)
        f_transitions_b = np.load(fp)
        f_transitions_e = np.load(fp)
    return nf_transitions_b, nf_transitions_e, f_transitions_b, f_transitions_e
