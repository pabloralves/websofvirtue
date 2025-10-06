"""
Contains the model and methods to run it
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def rand_choice_nb(arr, prob):
    # NP Random choice not available with jit
    # Workaround from https://github.com/numba/numba/issues/2539
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@jit(nopython=True)
def moral_operation(dA, dB, m1, m0):
    individual_p_moral_A = dA * (m1 - m0) + m0
    individual_p_moral_B = dB * (m1 - m0) + m0
    return individual_p_moral_A, individual_p_moral_B

@jit(nopython=True)
def peer_influenceability(dA, dB, theta_x, theta_y):
    if dA <= theta_x:
        peer_influenceability_A = (theta_y / theta_x) * dA
    else:
        peer_influenceability_A = -theta_y / (1 - theta_x) * (dA - theta_x) + theta_y

    if dB <= theta_x:
        peer_influenceability_B = (theta_y / theta_x) * dB
    else:
        peer_influenceability_B = -theta_y / (1 - theta_x) * (dB - theta_x) + theta_y

    return peer_influenceability_A, peer_influenceability_B

#@jit
def run_model_n_times(runs,timesteps,model,network,development):

    # Variables from dict
    m0 = model['m0']
    m1 = model['m1']
    m12 = model['m12']
    m23 = model['m23']
    sx = model['sx']
    sy = model['sy']
    theta_x = model['theta_x']
    theta_y = model['theta_y']

    # Number of nodes
    N = network.shape[0]

    # Initial distribution
    node_moral_development = np.zeros((runs,timesteps,N))
    node_moral_development[:,0,:] =  development

    # Network topology
    node_adjacency_matrix = np.zeros((runs,timesteps,N,N))
    node_adjacency_matrix[:,0,:,:] = network

    # 2.1 Nodes
    nodes_ordered = np.arange(0, N)

    # Node random interaction order
    nodes = nodes_ordered.copy()
    node_interaction_order = np.zeros((runs,timesteps,N),dtype=np.int64)

    for r in range(runs):
        for t in range(timesteps):
            np.random.shuffle(nodes)
            node_interaction_order[r,t,:] = nodes

    # Logging: matrices
    logging_node_stages = np.zeros((runs,timesteps,N),dtype=np.int32)
    logging_node_stages_grouped = np.zeros((runs,timesteps,3),dtype=np.int32)
    logging_node_stages_averaged = np.zeros((runs,timesteps))
    logging_average_morality = np.zeros((runs,timesteps))
    logging_social_bias = np.zeros((runs,timesteps))

    # Logging: node B
    logging_B = np.zeros((runs,timesteps,N))
    logging_B_contr_individual = np.zeros((runs,timesteps,N))
    logging_B_contr_group = np.zeros((runs,timesteps,N))
    logging_B_contr_society = np.zeros((runs,timesteps,N))
    logging_B_m_final = np.zeros((runs,timesteps,N))
    logging_B_m_facto = np.zeros((runs,timesteps,N))

    logging_node_neighborhood_A_weighted_moralities_average = np.zeros((runs,timesteps,N))
    logging_neighborhood_B_weighted_moralities_average = np.zeros((runs,timesteps,N))
    
    logging_node_m_contribution_individual = np.zeros((runs,timesteps,N))
    logging_node_m_contribution_group = np.zeros((runs,timesteps,N))
    logging_node_m_contribution_society = np.zeros((runs,timesteps,N))
    logging_node_m_final = np.zeros((runs,timesteps,N))
    logging_node_m_facto = np.zeros((runs,timesteps,N))

    ####################
    # 3. RUN THE MODEL #
    ####################
    for R in range(runs):

        # Logging for initial timestep
        logging_average_morality[R,0] = np.mean(node_moral_development[R,0,:])
        logging_node_stages[R,0] = np.where(node_moral_development[R,0] <= m12, 0,np.where(node_moral_development[R,0] <= m23, 1, 2))
        logging_node_stages_grouped[R,0,0] = np.sum(logging_node_stages[R,0,:] == 0)
        logging_node_stages_grouped[R,0,1] = np.sum(logging_node_stages[R,0,:] == 1)
        logging_node_stages_grouped[R,0,2] = np.count_nonzero(logging_node_stages[R,t,:] == 2)
        logging_node_stages_averaged[R,0] = np.mean(logging_node_stages[R,0,:])/2
        
        society_average_morality = logging_average_morality[R,0]

        if society_average_morality <= sx:
            logging_social_bias[R,0] = -sy
        elif society_average_morality > (1-sx):
            logging_social_bias[R,0] = sy
        else:
            logging_social_bias[R,0] = (2*sy)/(1-2*sx)*(society_average_morality - sx) - sy

        for t in range(timesteps-1):
            for A in node_interaction_order[R,t,:]:
                ######################################################################################################
                # 1. Get agent B to interact with
                ######################################################################################################

                # 1.1 Ensure agent can't bond with itself)
                node_adjacency_matrix[R,t,A,A] = 0

                # 1.2 If A has no bonds, pick a random node (except itself)
                if np.all(node_adjacency_matrix[R,t,A,:] == 0):
                    node_adjacency_matrix[R,t,A,:] = 0.01
                    node_adjacency_matrix[R,t,A,A] = 0 

                # 1.3 Get agent B probabilistically based on the bond strength
                node_idx_sum = np.array(np.sum(node_adjacency_matrix[R,t,A,:]))
                p_interact_with_other_nodes = np.array(node_adjacency_matrix[R,t,A,:] / node_idx_sum)
                B = rand_choice_nb(nodes_ordered, p_interact_with_other_nodes) #interact_with_other_nodes(node_adjacency_matrix, R, t, A, nodes_ordered)

                ######################################################################################################
                # 2. Get interaction morality
                ######################################################################################################                
                dA = node_moral_development[R,t,A]
                dB = node_moral_development[R,t,B]

                individual_p_moral_A, individual_p_moral_B = moral_operation(dA, dB, m1, m0)
                peer_influenceability_A, peer_influenceability_B = peer_influenceability(dA, dB, theta_x, theta_y)

                # 2.2 Individual-level contributors to act morality
                A_individual_term = individual_p_moral_A * (1-peer_influenceability_A)
                B_individual_term = individual_p_moral_B * (1-peer_influenceability_B)

                # 2.3 Group-level contributors to act morality
                neighborhood_moralities = node_moral_development[R,t,:]*(m1-m0) + m0                                                

                neighborhood_A = node_adjacency_matrix[R,t,A,:]         
                neighborhood_A_weighted_moralities = neighborhood_A * neighborhood_moralities
                neighborhood_A_weighted_moralities_average = np.sum(neighborhood_A_weighted_moralities)/(N-1)

                neighborhood_B = node_adjacency_matrix[R,t,B,:]                                                                 
                neighborhood_B_weighted_moralities = neighborhood_B * neighborhood_moralities
                neighborhood_B_weighted_moralities_average = np.sum(neighborhood_B_weighted_moralities)/(N-1)   # Do not count node A for the average

                A_group_bias_term = neighborhood_A_weighted_moralities_average * peer_influenceability_A        # Stage independent! No in-out group differences
                B_group_bias_term = neighborhood_B_weighted_moralities_average * peer_influenceability_B        # Stage independent! No in-out group differences

                # 2.4 Social-level term
                society_average_morality = np.mean(neighborhood_moralities)

                if society_average_morality <= sx:
                    AB_social_bias_term = -sy
                elif society_average_morality > (1-sx):
                    AB_social_bias_term = sy
                else:
                    AB_social_bias_term = (2*sy)/(1-2*sx)*(society_average_morality - sx) - sy

                # 2.5 Combine the individual, group and societal level contributors to act morality
                A_morality = A_individual_term + A_group_bias_term + AB_social_bias_term
                B_morality = B_individual_term + B_group_bias_term + AB_social_bias_term
                A_morality = np.clip(A_morality,0,1)
                B_morality = np.clip(B_morality,0,1)

                ######################################################################################################
                # 3. Get de-facto morality
                ######################################################################################################

                # 3.1 Run Bernoulli trials with the aggregated probability of acting morally
                # To determine whether the agents act morally or not in the current interaction
                A_moral = 1 if np.random.rand() <= A_morality else 0
                B_moral = 1 if np.random.rand() <= B_morality else 0

                ######################################################################################################
                # 4. Develop agents morally based on their stage and de-facto morality
                ######################################################################################################                
                development_A_sign = 1 if B_moral else -1

                size_development_A = 0.01
                size_development_B = 0.01
                
                # 4.4 Compute the final change in development for the agents
                delta_development_A = size_development_A*development_A_sign
                
                # 4.3 Use this change in moral development to update the individual development
                node_moral_development[R,t+1,A] = np.clip(node_moral_development[R,t,A] + delta_development_A,0,1)
                
                # 4.4 LOGGING
                logging_B[R,t,A] = B
                logging_node_m_contribution_individual[R,t,A] = A_individual_term
                logging_node_m_contribution_group[R,t,A] = A_group_bias_term
                logging_node_m_contribution_society[R,t,A] = AB_social_bias_term
                logging_node_m_final[R,t,A] = A_morality
                logging_node_m_facto[R,t,A] = A_moral
                
                logging_node_neighborhood_A_weighted_moralities_average[R,t,A] = neighborhood_A_weighted_moralities_average
                logging_neighborhood_B_weighted_moralities_average[R,t,A] = neighborhood_B_weighted_moralities_average

                logging_B_contr_individual[R,t,A] = B_individual_term
                logging_B_contr_group[R,t,A] = B_group_bias_term
                logging_B_contr_society[R,t,A] = AB_social_bias_term
                logging_B_m_final[R,t,A] = B_morality
                logging_B_m_facto[R,t,A] = B_moral
                
            # Update network topology for next step
            node_adjacency_matrix[R,t+1,:,:] = node_adjacency_matrix[R,t,:,:]

            ######################################################################################################
            # 6. Log relevant variables for analysis after each timestep
            ######################################################################################################
            # 6.1 Average morality
            logging_average_morality[R,t+1] = np.mean(node_moral_development[R,t,:])

            # 6.1 Agent stage (both individual and aggregated)
            logging_node_stages[R,t+1,:] = np.where(node_moral_development[R,t] <= m12, 0,np.where(node_moral_development[R,t] <= m23, 1, 2))
            logging_node_stages_grouped[R,t+1,0] = np.sum(logging_node_stages[R,t,:] == 0)
            logging_node_stages_grouped[R,t+1,1] = np.sum(logging_node_stages[R,t,:] == 1)
            logging_node_stages_grouped[R,t+1,2] = np.count_nonzero(logging_node_stages[R,t,:] == 2)
            logging_node_stages_averaged[R,t+1] = np.mean(logging_node_stages[R,t,:])/2

            # 6.2 Social bias
            society_average_morality = np.mean(logging_average_morality[R,t+1])
            
            if society_average_morality <= sx:
                logging_social_bias[R,t+1] = -sy
            elif society_average_morality > (1-sx):
                logging_social_bias[R,t+1] = sy
            else:
                logging_social_bias[R,t+1] = (2*sy)/(1-2*sx)*(society_average_morality - sx) - sy


     # Create dictionary with results
    results = {
        # Macro variables (timestep)
        "nodes" : nodes_ordered,
        "node_development" : node_moral_development,
        "adjacency_matrix" : node_adjacency_matrix,
        "node_stages" : logging_node_stages,
        "node_stages_grouped" : logging_node_stages_grouped,
        "node_stages_averaged" : logging_node_stages_averaged,
        "average_morality" : logging_average_morality,
        "social_bias" : logging_social_bias,
        # Micro interaction (micro-timestep)
        "logging_node_interacted_with": logging_B,
        "logging_node_m_contribution_individual": logging_node_m_contribution_individual,
        "logging_node_m_contribution_group":logging_node_m_contribution_group,
        "logging_node_m_contribution_society":logging_node_m_contribution_society,
        "logging_node_m_final":logging_node_m_final,
        "logging_node_m_facto":logging_node_m_facto,
        "logging_node_neighborhood_A_weighted_moralities_average":logging_node_neighborhood_A_weighted_moralities_average,
        "logging_neighborhood_B_weighted_moralities_average":logging_neighborhood_B_weighted_moralities_average,
        "logging_node_interacted_with_contr_individual": logging_B_contr_individual,
        "logging_node_interacted_with_contr_group":logging_B_contr_group,
        "logging_node_interacted_with_contr_society":logging_B_contr_society,
        "logging_node_interacted_with_m_final":logging_B_m_final,
        "logging_node_interacted_with_m_facto":logging_B_m_facto,
    }

    return results