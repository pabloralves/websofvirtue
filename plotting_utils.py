"""
Functions to plot inputs and outputs
"""

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from experiment_utils import get_individual_morality,get_susceptibility,get_society_bias,get_agent_stage


def plot_model_summary(model,points_plotting):

    #sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, sharex='all',  figsize=(8, 3))
    x = np.linspace(0,1,points_plotting)
    y_individual = np.zeros(points_plotting)
    y_group = np.zeros(points_plotting)
    y_society = np.zeros(points_plotting)

    for i in range(points_plotting):
        y_individual[i] = get_individual_morality(x[i],model['m0'],model['m1'])
        y_group[i] = get_susceptibility(x[i],model['theta_x'],model['theta_y'])
        y_society[i] = get_society_bias(x[i],model['sx'],model['sy'])

    axes[0].plot(x,y_individual,label='$m(d)$',lw=2,zorder=3,c='red')
    axes[0].plot(x,y_group,label='$\Theta(d)$',lw=2,zorder=3,c='tab:blue')
    axes[1].plot(x,y_society,label='$S(\mu_s)$',lw=2,zorder=3,c='tab:orange')

    axes[0].set_title(f"Individual-level factors\n$m_0$={model['m0']}, $m_1$={model['m1']}, $\Theta_x$={ model['theta_x']}, $\Theta_y$={model['theta_y']}")
    axes[1].set_title(f"Social-bias\n$S_x$={model['sx']}, $S_y$={ model['sy']}")

    axes[0].axhline(y=model['theta_y'], xmax=1, color='grey', linestyle='--',zorder=2)#,label='$\Theta_y$')
    axes[0].axvline(x=model['theta_x'], ymin=0, ymax=1, color='grey', linestyle='--',zorder=2)#,label='$\Theta_x$')
    axes[1].axvline(x=model['sx'], color='grey', linestyle='--',zorder=2)#,label='$S_x$')
    axes[1].axvline(x=1-model['sx'], color='grey', linestyle='--',zorder=2)#,label='$S_x$')
 #   axes[1].axhline(y=GLOBAL_SY, color='black', linestyle='--',zorder=2)#,label='$S_y$')
#    axes[1].axhline(y=-GLOBAL_SY, color='black', linestyle='--',zorder=2)#,label='$S_y$')


    axes[0].set_ylim([0,1.1])
    axes[1].set_xticks(np.arange(0,1,0.1))
    axes[1].set_xlim([0,1])
    axes[1].set_ylim([-model['sy']-0.1,model['sy']+0.1])    
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel('$d$')
    axes[1].set_xlabel('$\overline{\mu_m}$')
    axes[1].set_ylabel('$\Xi(\overline{\mu_m})$')
    plt.tight_layout()

    # Indirect parameter
    idx = np.argwhere(np.diff(np.sign(y_group - y_individual))).flatten()
    axes[0].plot(x[idx], y_group[idx], 'ro',zorder=4)

    # Egocentric to Ethnocentric / Ethnocentric to Worldcentric
    M12 = x[idx][0]
    M23 = x[idx][1]

    axes[0].grid(axis='both')    
    plt.grid()
    axes[1].grid(axis='both')
    plt.grid()

    plt.show()



def plot_network_development_summary(model,network,development,adjacency,color=True,dpi=100):
    """
    Plots the given network and its level of development in the population
    """


    if color:
        A = adjacency
        node_stages_now = [get_agent_stage(n,model['m12'],model['m23']) for n in development]
        node_list = np.arange(adjacency.shape[0])
        G = nx.from_numpy_array(A, parallel_edges=False, create_using=None, edge_attr='stage', nodelist=node_list)
        values = dict(zip(node_list, node_stages_now))
        nx.set_node_attributes(G, values = values, name='stage')
        nodes_ego = [n for (n,ty) in nx.get_node_attributes(G,'stage').items() if ty == 0]
        nodes_etno = [n for (n,ty) in nx.get_node_attributes(G,'stage').items() if ty == 1]
        nodes_world = list(set(G.nodes()) - set(nodes_etno) - set(nodes_ego))
        pos = nx.spring_layout(G)

        #plt.figure(figsize=(6,2))
        nx.draw_networkx(G, pos, with_labels=False, node_color='black', node_size=10, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_ego, node_color='red', node_shape='o')
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_etno, node_color='orange', node_shape='o')
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_world, node_color='green', node_shape='o')
        labels = dict([(n, n) for n in G.nodes()])
        nx.draw_networkx_labels(G, pos, labels=labels,font_color='white')
        plt.show()
    else:  
        G = network
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', 
                        node_size=500, edge_color='gray')

        labels = dict([(n, n) for n in G.nodes()])
        nx.draw_networkx_labels(G, pos, labels=labels)
        plt.show()
    

    #########################################################3

    # Plot histogram with distribution of stages
    agents_s1 = len(development[development < model['m12']])
    agents_s2 = len(development[(development >= model['m12']) & (development <= model['m23'])])
    agents_s3 = len(development[(development > model['m23'])])

    # Barplot
    # See https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_colors.html#sphx-glr-gallery-lines-bars-and-markers-bar-colors-py
    fig, ax = plt.subplots(figsize=(6,2))
    stages = ['$S_1$', '$S_2$', '$S_3$']
    counts = [agents_s1/len(network.nodes), agents_s2/len(network.nodes), agents_s3/len(network.nodes)]
    #bar_labels = [str(agents_s1)+str(counts[0]),str(agents_s1)+str(counts[1]),str(agents_s1)+str(counts[2])]
    bar_labels = [f"{agents_s1} ({counts[0]*100}%)",f"{agents_s2} ({counts[1]*100}%)",f"{agents_s3} ({counts[2]*100}%)"]
    bar_colors = ['red', 'tab:orange', 'tab:green']

    ax.bar(stages, counts, label=bar_labels, color=bar_colors)

    ax.set_ylabel('Fraction of agents')
    ax.set_title('Developmental stage distribution of population')
    ax.legend()
    plt.grid()
    plt.show()

    #print(agents_s1,agents_s2,agents_s3)

    #TODO: Plot stage-distribution
    #TODO: Plot statistics of network

    pass




def plot_stage_distribution(node_stages_grouped,average_stage_counts,average_morality,nodes,figsize=(6,4),dpi=100):
    # Seaborn plots
    average_stage_counts = np.mean(node_stages_grouped, axis=0)
    average_stage_counts = average_stage_counts/nodes
    std_stage_counts = np.std(node_stages_grouped, axis=0)/nodes


    sns.set_style("whitegrid")

    plt.figure(figsize=figsize,dpi=dpi)

    plt.plot(average_stage_counts[:,0],label='$s_1$',c='red')
    plt.plot(average_stage_counts[:,1],label='$s_2$',c='tab:orange')
    plt.plot(average_stage_counts[:,2],label='$s_3$',c='tab:green')

    plt.fill_between(np.arange(len(average_stage_counts)), average_stage_counts[:, 0] - std_stage_counts[:, 0], average_stage_counts[:, 0] + std_stage_counts[:, 0], alpha=0.3, edgecolor='none',color='red')
    plt.fill_between(np.arange(len(average_stage_counts)), average_stage_counts[:, 1] - std_stage_counts[:, 1], average_stage_counts[:, 1] + std_stage_counts[:, 1], alpha=0.2, edgecolor='none',color='tab:orange')
    plt.fill_between(np.arange(len(average_stage_counts)), average_stage_counts[:, 2] - std_stage_counts[:, 2], average_stage_counts[:, 2] + std_stage_counts[:, 2], alpha=0.2, edgecolor='none',color='tab:green')

    average_morality_sim = np.mean(average_morality,axis=0)
    plt.plot(average_morality_sim,label='$\mu_m$',c='tab:blue',ls='--')
    #plt.plot(average_morality_sim,label='$\mu_m$',c='tab:blue')


    plt.ylim([0,1])
    plt.xlabel('Timestep')
    plt.ylabel('Fraction nodes')
    plt.title('Developmental stage distribution')
    plt.legend()
    plt.show()





def plot_development(node_list,node_development,model_m12,model_m23,dpi=100):
    network_nodes = len(node_list)
    starting_node_stages = np.array([get_agent_stage(n,model_m12,model_m23) for n in node_development])
    #plt.bar(nodes_ego,development_matrix_0[nodes_ego],label='$m_{0,i}$')
    plt.figure(figsize=(8,3),dpi=dpi)
    plt.bar(node_list[np.where(starting_node_stages==0)],node_development[np.where(starting_node_stages==0)],label='$s_1$',color='red')
    plt.bar(node_list[np.where(starting_node_stages==1)],node_development[np.where(starting_node_stages==1)],label='$s_2$',color='tab:orange')
    plt.bar(node_list[np.where(starting_node_stages==2)],node_development[np.where(starting_node_stages==2)],label='$s_3$',color='tab:green')
    plt.axhline(y=model_m12,ls='--',c='grey',label='$m_{12}=$'+str(round(model_m12,3)))
    plt.axhline(y=model_m23,ls='--',c='grey',label='$m_{23}=$'+str(round(model_m23,3)))
    plt.ylim([0,1])
    plt.xlabel('Node')
    plt.ylabel('Development')
    plt.xticks(np.arange(network_nodes))
    plt.title('Agent development')
    plt.legend()
    plt.grid()
    plt.show()


# Number of neighbors

def plot_neighbors(node_list,adjacency_matrix):
    network_nodes = len(node_list)
    starting_node_neigbors = np.array([np.sum(row) for row in adjacency_matrix])
    plt.figure(figsize=(8,3))
    plt.bar(node_list,starting_node_neigbors,color='tab:blue')
    plt.xlabel('Node')
    plt.ylabel('Neighbors')
    plt.xticks(np.arange(network_nodes))
    plt.title('Agent neighbors in network')
    plt.grid()
    plt.show()