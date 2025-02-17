import os
import utils as ut
import pandas as pd
import seaborn as sns
import networkx as nx
import multiprocessing
sns.set_style("darkgrid")
from copy import deepcopy
from heapq import nlargest
from generalGreedy import *
from config import infMaxConfig
# from CCparallel import CC_parallel
#import matplotlib.pylab as plt
# from load_facebook_rice import *
# from load_facebook_graph import *
# from load_instagram import *
import matplotlib.pyplot as plt
# from load_twitter_pol_data import *
from IC import runIC, avgSize, runIC_fair_timings
from generateGraph import generateGraphNPP, generate_example

class fairInfMaximization(infMaxConfig):

    def __init__(self):
        super(fairInfMaximization,self).__init__()

        if self.synthetic1: 
            filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}_{self.weight}'
            #print(self.p_with, self.p_across)
            self.G = ut.load_graph(filename, self.p_with, self.p_across,  self.group_ratio , self.num_nodes, self.weight)

       
        elif self.facebook:
            filename = self.filename + f'_communities'
            self.G, num_groups = facebook_circles_graph(self.filename, filename, self.num_groups, self.weight) #ut.get_facebook_data(filename, w = self.weight)
            print(f'num groups: {num_groups}')

        elif self.facebook_rice:
            self.G, self.num_groups = facebook_rice(self.filename, self.filename_communities , self.weight) #ut.get_facebook_data(filename, w = self.weight)
            print(f'num groups: {self.num_groups}')


        elif self.instagram: 
            self.G, self.num_groups = instagram(self.filename, self.filename_communities , self.weight) #ut.get_facebook_data(filename, w = self.weight)
            self.filename = self.filename + f'_{self.weight}_'
            print(f'num groups: {self.num_groups}')

        self.stats = ut.graph_stats(self.G, self.num_groups)

      
    def run_root_formulation(self):


        if self.synthetic1:
            filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}_{self.weight}'
        else:
            filename = self.filename +f'_{self.weight}'
        
        labels = []
        for gamma in self.gammas_root:
                print("power root ", gamma)
                # for beta in self.beta_root:

                    # rlg = reduced_lazy_greedy(self.G)
                    # influenced_grouped, seeds_grouped = rlg.lazyGreedy_node_parallel_pre_built_graphs(filename, self.G, self.seed_size, self.tau, type_algo = 3, num_groups = self.num_groups, batch_size = self.batch_size, fact = self.cpu_process_factor)

                influenced_grouped, seeds_grouped = lazyGreedy_node_parallel(filename, self.G, self.seed_size, self.tau, gamma = gamma ,  type_algo = 3, num_groups = self.num_groups, batch_size = self.batch_size, fact = self.cpu_process_factor)
                    
        return influenced_grouped,seeds_grouped
                 #ut.plot_influence(influenced_a, influenced_b, self.seed_size , filename , self.stats['group_a'], self.stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
        #             influenced_a_list.append(influenced_a)
        #             influenced_b_list.append(influenced_b)
        #         labels.append(f'gamma = {gamma}')
        # influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename)
        # influenced_a_list.append(influenced_a)
        # influenced_b_list.append(influenced_b)
        # labels.append("Greedy")
        # filename = "results/comparison_root_"
        # ut.plot_influence_diff(influenced_a_list, influenced_b_list, self.seed_size, labels, filename, self.stats['group_a'], self.stats['group_b'] ) 

    def run_log_formulation(self):

        if self.synthetic1:
            filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}_{self.weight}'
        else:
            filename = self.filename + f'_{self.weight}'

        labels =  []
        # for gamma in self.gammas_log:

                    # rlg = reduced_lazy_greedy(self.G)
                    # influenced_grouped, seeds_grouped = rlg.lazyGreedy_node_parallel_pre_built_graphs(filename, self.G, self.seed_size, self.tau, type_algo = 2, num_groups = self.num_groups, batch_size = self.batch_size, fact = self.cpu_process_factor)

        influenced_grouped, seeds_grouped = lazyGreedy_node_parallel(filename, self.G, self.seed_size,  self.tau, type_algo = 2, num_groups = self.num_groups,batch_size = self.batch_size, fact = self.cpu_process_factor)

                    #influenced_grouped, seeds_grouped = generalGreedy_node_parallel(filename, self.G, self.seed_size,  self.tau,gamma = gamma, type_algo = 2, num_groups = self.num_groups)
                    
                    #ut.plot_influence(influenced_a, influenced_b, self.seed_size , filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
                    #influenced_a_list.append(influenced_a)
                    #influenced_b_list.append(influenced_b)
                    #labels.append(f'gamma = {gamma}')
        #influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename)
        #influenced_a_list.append(influenced_a)
        #influenced_b_list.append(influenced_b)
        #labels.append("Greedy")
        #filename = "results/comparison_log_"
        #ut.plot_influence_diff(influenced_a_list, influenced_b_list, self.seed_size, labels, filename, self.stats['group_a'], self.stats['group_b'] )             
        return influenced_grouped, seeds_grouped

    def run_set_cover_formulation_no_groups(self):

        print ( f"**** Running unfair Cover {self.reach}")
        if self.synthetic1:
            filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}_{self.weight}'
            # reach = self.reach #reach_list[0]
            influenced_grouped, seeds_grouped = generalGreedy_node_set_cover(filename, self.G, self.reach, self.tau, type_algo = 1, num_groups = self.num_groups)
        else:
            filename = self.filename + f'_{self.weight}'
            influenced_grouped, seeds_grouped = lazyGreedy_node_set_cover(filename, self.G, self.reach, self.tau, type_algo = 1, num_groups = self.num_groups, budget_range = self.reach_list, fact = self.cpu_process_factor, batch_size = self.batch_size)
        
        #for reach in self.reach_list:
        # reach = self.reach #reach_list[0]
        #influenced_grouped, seeds_grouped = generalGreedy_node_set_cover(filename, self.G, self.reach_list[0], self.tau, type_algo = 1, num_groups = self.num_groups)
        
        #rlg = reduced_lazy_greedy(self.G)
        #influenced_grouped, seeds_grouped = rlg.lazyGreedy_node_set_cover_pre_built_graphs(filename, self.G, reach, self.tau, type_algo = 1, num_groups = self.num_groups, budget_range = self.reach_list, fact = self.cpu_process_factor, batch_size = self.batch_size)
        #lazyGreedy_node_set_cover(filename, self.G, reach, self.tau, type_algo = 1, num_groups = self.num_groups, budget_range = self.reach_list, fact = self.cpu_process_factor, batch_size = self.batch_size)
        return influenced_grouped, seeds_grouped


    def run_set_cover_formulation(self):

        print ( f"**** Running fair Cover {reach}")
        if self.synthetic1:
            filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}_{self.weight}'
            influenced_grouped, seeds_grouped = generalGreedy_node_set_cover(filename, self.G, self.reach, self.tau, type_algo = 2, num_groups = self.num_groups)
        else:
            filename = self.filename + f'_{self.weight}'
            influenced_grouped, seeds_grouped = lazyGreedy_node_set_cover(filename, self.G, self.reach, self.tau, type_algo = 2, num_groups = self.num_groups, fact = self.cpu_process_factor, batch_size = self.batch_size)
        
        #for reach in self.reach_list:

        #influenced_grouped, seeds_grouped = generalGreedy_node_set_cover(filename, self.G, 0.1, self.tau, type_algo = 2, num_groups = self.num_groups)
        # reach = self.reach #reach_list[0]
        #influenced_grouped, seeds_grouped = generalGreedy_node_set_cover(filename, self.G, self.reach_list[0], self.tau, type_algo = 1, num_groups = self.num_groups)
        

        #rlg = reduced_lazy_greedy(self.G)
        #influenced_grouped, seeds_grouped = rlg.lazyGreedy_node_set_cover_pre_built_graphs(filename, self.G, reach, self.tau, type_algo = 2, num_groups = self.num_groups, budget_range = self.reach_list, fact = self.cpu_process_factor, batch_size = self.batch_size)
        
        return influenced_grouped, seeds_grouped
            #ut.plot_influence(influenced_a, influenced_b, self.seed_size , filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
            # influenced_a_list.append(influenced_a)
            # influenced_b_list.append(influenced_b)

                  
    def calculate_greedy(self, fn = None, G = None):
        '''
        returns greedy algorithms's inf , inf_a , inf_b, seeds_a, seeds_b ( at each iteration and iteration being at each seed selection)
        '''
        if G == None: 
            G = self.G

        if self.synthetic1:
            filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}_{self.weight}'
        else:
            filename = self.filename + f'_{self.weight}_'


        return generalGreedy_node_parallel(filename, G, self.seed_size, self.tau, type_algo = 1, num_groups = self.num_groups)

    def calculate_lazygreedy(self, fn = None, G = None):
        '''
        returns greedy algorithms's inf , inf_a , inf_b, seeds_a, seeds_b ( at each iteration and iteration being at each seed selection)
        '''
        if G == None: 
            G = self.G

        if self.synthetic1:
            filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}_{self.weight}'
        else:
            filename = self.filename + f'_reduced_{self.weight}'

        #rlg = reduced_lazy_greedy(G)
        #return rlg.lazyGreedy_node_parallel_pre_built_graphs(filename, G, self.seed_size, self.tau, type_algo = 1, num_groups = self.num_groups, batch_size = self.batch_size, fact = self.cpu_process_factor)

        return lazyGreedy_node_parallel(filename, self.G, self.seed_size, self.tau, type_algo = 1, num_groups = self.num_groups, batch_size = self.batch_size, fact = self.cpu_process_factor)

    def effect_of_group_sizes(self):
        '''
        This generate the evaluation graphs for 
        
        ii) varrying p_g_a
        '''
        influenced_a_list = []
        influenced_b_list = []
        
        for group_ratio in self.group_ratios:
        #group_ratio = 0.5 #0.7 
            # A loop here to run multiple times on 5 seeds 
            # for seed in SEED_list:
            filename=f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{group_ratio}'
            

            # read in graph
            G = ut.load_graph(filename, self.p_with, self.p_across,  group_ratio ,self.num_nodes)
            
            
            
            influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename, G)

            stats = ut.graph_stats(G, print_stats = True)
            
            influenced_a_list.append(influenced_a)
            influenced_b_list.append(influenced_b)
            seeds_a_list.append(seeds_a)
            seeds_b_list.append(seeds_b)

        print( " ******* Finished group size analysis *******")

        return (influenced_a_list,influenced_b_list, seeds_a_list, seeds_b_list)

    def effect_of_across_group_connectivity(self):
        '''
        This generate the evaluation graphs for 
        i) varrying p_across with p_g_a = 0.5 
        
        '''
        # Have to do this for multiple runs, and or multiple graphs 
        influenced_a_list = []
        influenced_b_list = []
        seeds_a_list = []
        seeds_b_list = []
        group_ratio = 0.5 # just to bring out the effect of p_across

        for p_across in self.p_acrosses:

           
            filename=f'{self.filename}_{self.num_nodes}_{self.p_with}_{p_across}_{group_ratio}'
            
            # read in graph
            G = ut.load_graph(filename, self.p_with, p_across, group_ratio ,self.num_nodes)
            
            influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename) # 

            stats = ut.graph_stats(G, print_stats = True)
     
            ut.plot_influence(influenced_a, influenced_b, self.seed_size, filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])

            influenced_a_list.append(influenced_a)
            influenced_b_list.append(influenced_b)
            seeds_a_list.append(seeds_a)
            seeds_b_list.append(seeds_b)

        print( " ******* Finished connectivity analysis *******")

        return (influenced_a_list, influenced_b_list, seeds_a_list, seeds_b_list)

        ## varies group sizes 

def save_fairness_data_maximization():
    
    fair_inf = fairInfMaximization()
    Influenced_root, seed_grouped_root = fair_inf.run_root_formulation()       

    fair_inf = fairInfMaximization()
    Influenced_log, seed_grouped_log = fair_inf.run_log_formulation()    

    fair_inf = fairInfMaximization()
    Influenced_greedy, seed_grouped_greedy = fair_inf.calculate_lazygreedy()
    #calculate_greedy(fair_inf.filename)

    if fair_inf.synthetic1:
            filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{fair_inf.weight}'
    else:
            filename = fair_inf.filename + f'_{fair_inf.weight}_'
            #filename = self.filename + f'_reduced_{self.weight}'

    poplulation_group = fair_inf.stats[0]
    #
    #
    #File is       total  g1   g2 
    #.      greedy  xx.    xx. xx. 
    #       log     xx.    xx. xx
    #       root    xx.   xx.   xx.  
    f = open(filename +'_fairness.txt', 'w')
    f.write(f'{str(np.sum(np.asarray(Influenced_greedy[-1])) / np.sum(np.asarray(poplulation_group)))}\t')
    for idx, I in enumerate(Influenced_greedy[-1]):
        f.write(f"{I/poplulation_group[idx]}\t")
    f.write(f'\n')

    f.write(f'{str(np.sum(np.asarray(Influenced_log[-1]))/ np.sum(np.asarray(poplulation_group)))}\t')
    for idx, I in enumerate(Influenced_log[-1]):
        f.write(f"{I/poplulation_group[idx]}\t")
    f.write(f'\n')
    f.write(f'{str(np.sum(np.asarray(Influenced_root[-1]))/ np.sum(np.asarray(poplulation_group)))}\t')
    for idx, I in enumerate(Influenced_root[-1]):
        f.write(f"{I/poplulation_group[idx]}\t")

    print(filename +'_fairness.txt')
    f.close()

    csv_file = pd.read_csv(filename +'_fairness.txt', sep = "\t", header = None)
    csv_file.dropna(axis = 'columns')
    xlabels = ["P1", "P4-Log", "P4-root"]
    barWidth = 0.3
    r1 = np.arange(len(xlabels))
    r2 = [x + barWidth for x in r1]
    r3= [x + barWidth for x in r2]


    plt.rcParams.update({'font.size': 12})
    # figure(figsize=(15, 6), dpi=80)
    plt.bar(r1,csv_file[0], color='blue',width=barWidth, label = "Total")
    plt.bar(r2,csv_file[1], color='orange',width=barWidth, label = "Group-1")
    plt.bar(r3,csv_file[2], color='green',width=barWidth, label = "Group-2")



    plt.xticks([r + barWidth for r in range(len(xlabels))], xlabels)   
    plt.legend(bbox_to_anchor=(0.05, 1.18),loc='upper left', fontsize=12, ncol=3)
    plt.ylabel("Fraction influenced")
    plt.savefig(f"{filename}_total_group_influence.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def save_group_maximization():

    fair_inf = fairInfMaximization()
    poplulation_group = fair_inf.stats[0]
    group_ratios = [0.5,0.55,0.6,0.7,0.8]
    f = open(fair_inf.filename + f'_group_analysis.txt', 'w')
    for g in group_ratios:

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{g}_{fair_inf.weight}_greedy_tau_{fair_inf.tau}_'
        else:
                filename = fair_inf.filename + f'_{fair_inf.weight}_greedy_tau_{fair_inf.tau}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)

        I = influenced_grouped[-1] 
        #for idx, I in enumerate(Influenced_greedy[-1]):
        g_a = int(g * fair_inf.num_nodes)
        g_b = fair_inf.num_nodes - g_a
        f.write(f"{abs(I[0]/g_a - I[1]/g_b )}\t") # assuming only two classes 
    

   

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{g}_{fair_inf.weight}_log_gamma_{fair_inf.gamma_log}_tau_{fair_inf.tau}_'
        else:
                filename = fair_inf.filename + f'_{fair_inf.weight}_log_gamma_{fair_inf.gamma_log}_tau_{fair_inf.tau}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)

        I = influenced_grouped[-1] 
        g_a = int(g * fair_inf.num_nodes)
        g_b = fair_inf.num_nodes - g_a
        #for idx, I in enumerate(Influenced_greedy[-1]):
        f.write(f"{abs(I[0]/g_a - I[1]/g_b)}\n") # assuming only two classes 
        
        #f.write(f'\n')

    f.close()

def save_clique_maximization():

    fair_inf = fairInfMaximization()
    poplulation_group = fair_inf.stats[0]
    p_acrosses = [0.025,0.015,0.01,0.001]
    f = open(fair_inf.filename + f'_clique_analysis.txt', 'w')
    for p in p_acrosses:

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{p}_{fair_inf.group_ratio}_{fair_inf.weight}_greedy_tau_{fair_inf.tau}_'
        else:
                filename = fair_inf.filename + f'_{fair_inf.weight}_greedy_tau_{fair_inf.tau}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)

        I = influenced_grouped[-1] 
        #for idx, I in enumerate(Influenced_greedy[-1]):
        #g_a = int(g * fair_inf.num_nodes)
        #g_b = fair_inf.num_nodes - g_a
        f.write(f"{abs(I[0]/poplulation_group[0] - I[1]/poplulation_group[1] )}\t") # assuming only two classes 
    

   

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{p}_{fair_inf.group_ratio}_{fair_inf.weight}_log_gamma_{fair_inf.gamma_log}_tau_{fair_inf.tau}_'
        else:
                filename = fair_inf.filename + f'_{fair_inf.weight}_log_gamma_{fair_inf.gamma_log}_tau_{fair_inf.tau}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)

        I = influenced_grouped[-1] 
        #g_a = int(g * fair_inf.num_nodes)
        #g_b = fair_inf.num_nodes - g_a
        #for idx, I in enumerate(Influenced_greedy[-1]):
        f.write(f"{abs(I[0]/poplulation_group[0] - I[1]/poplulation_group[1])}\n") # assuming only two classes 
        
        #f.write(f'\n')

    f.close()

def save_fairness_cost_cover():

    fair_inf = fairInfMaximization()
    poplulation_group = fair_inf.stats[0]
    # reach_list = [0.1]#,0.002]
    f= open(fair_inf.filename + f'_cover_fairness.txt', 'w')
    f_2= open(fair_inf.filename + f'_cover_cost.txt', 'w')
    # Cost file 
    #File is       unfair  fair    
    #.      r_1     xx.    xx.     
    #       r_2     xx.    xx. 
    #       r_3     xx.    xx. 
    # Influence file 
    #File is       unfair_g1 unfair_g2 ..  fair_g2 fair_g2    
    #.      r_1     xx.    xx.              xx.        xx.     
    #       r_2     xx.    xx.              xx.        xx. 
    #       r_3     xx.    xx.              xx.        xx.

    for r in fair_inf.reach_list:

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{fair_inf.weight}_set_cover_reach_{r}_tau_{fair_inf.tau}_no_groups_'
        else:
                #filename = fair_inf.filename + f'_set_cover_reach_{r}_tau_{fair_inf.tau}_no_groups_'
                filename = fair_inf.filename + f'_{fair_inf.weight}'

        results_exist, _, _,influenced_grouped, seeds_grouped = ut.get_results_cover(filename, fair_inf.G, r, fair_inf.tau, 1, fair_inf.num_groups) # algo type 1 i.e. unfair 
        if results_exist:
            print( f'found: {filename}')
        #fair_inf = fairInfMaximization()
        #influenced_grouped, seeds_grouped = fair_inf.run_set_cover_formulation_no_groups()
        cost = 0
        for S in seeds_grouped[-1]:
            cost += len(S)
        f_2.write(f"{cost}\t")
        for idx, I in enumerate(influenced_grouped[-1]):
            f.write(f"{I/poplulation_group[idx]}\t")

        # if r == 0.002: 
        #     continue 

        # if fair_inf.synthetic1:
        #         filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{fair_inf.weight}_set_cover_reach_{r}_tau_{fair_inf.tau}_'
        # else:
        #         filename = fair_inf.filename + f'_set_cover_reach_{r}_tau_{fair_inf.tau}_'

        results_exist, _,_, influenced_grouped, seeds_grouped = ut.get_results_cover(filename, fair_inf.G, r, fair_inf.tau, 2, fair_inf.num_groups)# algo type 2 i.e. fair 

        if results_exist:
            print( f'found: {filename} fair version ')

        #influenced_grouped, seeds_grouped = fair_inf.run_set_cover_formulation()
        
        cost = 0
        for S in seeds_grouped[-1]:
            cost += len(S)
        f_2.write(f"{cost}\n")

        for idx, I in enumerate(influenced_grouped[-1]):
            #if idx == 2:
            #    break
            f.write(f"{I/poplulation_group[idx]}\t")

        f.write("\n")

    f.close()
    f_2.close()

def save_activation_probability():
    '''
    This function assumes that the results for each activation probablity are already calculated. It just integrates them in one file 
    '''
    fair_inf = fairInfMaximization()
    poplulation_group = fair_inf.stats[0]
    weights = [0.01,0.05,0.1,0.2,0.3,1.0]
    f= open(fair_inf.filename + f'_budget_activation_tau_500.txt', 'w')
    for w in weights:

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{w}_greedy_tau_{500}_'
        else:
                filename = fair_inf.filename + f'_{w}_greedy_tau_{500}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)
        I = influenced_grouped[-1]
        disparity = abs(I[0]/poplulation_group[0] - I[1]/poplulation_group[1])
        f.write(f'{w}\t{disparity}\t')

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{w}_log_gamma_{fair_inf.gamma_log}_tau_{500}_'
        else:
                filename = fair_inf.filename + f'_{w}_log_gamma_{fair_inf.gamma_log}_tau_{500}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)
        I = influenced_grouped[-1]
        disparity = abs(I[0]/poplulation_group[0] - I[1]/poplulation_group[1])
        f.write(f'{disparity}\n')

    f.close()

    weights = [0.01,0.05,0.1,0.2,0.3,0.5,0.7,1.0]

    f= open(fair_inf.filename + f'_budget_activation_tau_2.txt', 'w')
    for w in weights:

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{w}_greedy_tau_{2}_'
        else:
                filename = fair_inf.filename + f'_{w}_greedy_tau_{2}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)
        I = influenced_grouped[-1]
        disparity = abs(I[0]/poplulation_group[0] - I[1]/poplulation_group[1])
        f.write(f'{w}\t{disparity}\t')

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{w}_log_gamma_{fair_inf.gamma_log}_tau_{2}_'
        else:
                filename = fair_inf.filename + f'_{w}_log_gamma_{fair_inf.gamma_log}_tau_{2}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)
        I = influenced_grouped[-1]
        disparity = abs(I[0]/poplulation_group[0] - I[1]/poplulation_group[1])
        f.write(f'{disparity}\n')

    f.close()

def save_time_deadline():

    '''
    Aggregates the already calculated results for different tau (time deadlines)
    '''
    fair_inf = fairInfMaximization()
    poplulation_group = fair_inf.stats[0]
    taus = [1,2,5,20,50,1300]
    f= open(fair_inf.filename + f'_budget_time_deadline.txt', 'w')
    for t in taus:

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{0.1}_greedy_tau_{t}_'
        else:
                filename = fair_inf.filename + f'_greedy_tau_{t}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)
        I = influenced_grouped[-1]
        disparity = abs(I[0]/poplulation_group[0] - I[1]/poplulation_group[1])
        f.write(f'{t}\t{disparity}\t')

        if fair_inf.synthetic1:
                filename = f'{fair_inf.filename}_{fair_inf.num_nodes}_{fair_inf.p_with}_{fair_inf.p_across}_{fair_inf.group_ratio}_{0.1}_log_gamma_{fair_inf.gamma_log}_tau_{t}_'
        else:
                filename = fair_inf.filename + f'_log_gamma_{fair_inf.gamma_log}_tau_{t}_'

        influenced_grouped, seeds_grouped = ut.read_files(filename)
        I = influenced_grouped[-1]
        disparity = abs(I[0]/poplulation_group[0] - I[1]/poplulation_group[1])
        f.write(f'{disparity}\n')

    f.close()

if __name__ == '__main__':

    import time
    start = time.time()

    # for saving to nicer formatted csvs after the results are calculated 
    # if False:
    #     save_time_deadline()

    # if False:
    #     save_activation_probability()

    # if False:
    #     save_fairness_cost_cover()

    # if False:
    #     save_clique_maximization()

    # if False:
    #     save_group_maximization()

    # For calculating results
    if False:
        # run the following to get the results for budge problem with and without fairness constraints. 
        save_fairness_data_maximization()

    if False:
        fair_inf = fairInfMaximization()
        fair_inf.effect_of_across_group_connectivity()
        fair_inf.effect_of_group_sizes()
           
    if False:
        # square root fairness formulation budget problem
        fair_inf = fairInfMaximization()
        fair_inf.run_root_formulation()

        
    if False:
        # log fairness formulation buget problem 
        fair_inf = fairInfMaximization()
        fair_inf.run_log_formulation()
        
    if False:
        # set cover with Fairness
        fair_inf = fairInfMaximization()
        for r in fair_inf.reach_list:
            fair_inf.reach = r
            fair_inf.run_set_cover_formulation()

    if True:
        # set cover without fairness 
        fair_inf = fairInfMaximization()
        for r in fair_inf.reach_list:
            fair_inf.reach = r
            fair_inf.run_set_cover_formulation_no_groups()

    if False:
        fair_inf = fairInfMaximization()
        #filename = fair_inf.filename + f'_weight_{fair_inf.weight}_'
        fair_inf.calculate_lazygreedy()#fair_inf.filename)

        # or also can use simple greedy 
        #fair_inf.calculate_greedy()

        
    print('Total time:', time.time() - start)
    
