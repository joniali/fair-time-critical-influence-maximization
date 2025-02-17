''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 1)
'''


from priorityQueue import PriorityQueue as PQ
from IC import *
import numpy as np
import multiprocessing
import utils as ut
import math 
import copy


def map_IC_timing(inp):

    G,S,v,gamma_a, gamma_b = inp
    R = 100
    priority = 0.0
    priority_a = 0.0
    priority_b = 0.0
    F_a = 0.0
    F_b = 0.0
    if v not in S:
        for j in range(R): # run R times Random Cascade
        # for different objective change this priority selection 
            T, T_a, T_b = runIC_fair_timings((G,S + [v], gamma_a, gamma_b))
            priority_a += float(T_a)/R
            priority_b += float(T_b)/R
            priority   += float(T_a + T_b)/R

    return (v,priority,priority_a, priority_b)

def map_IC(inp):
        G,S,p = inp
        #print(S)
        return len(runIC(G,S,p))

def map_fair_IC(inp):
    '''
    returns influence and list of different groups influenced
    '''
    G,S,num_groups,tau = inp
    #print(S)
    R = 500
    influenced = 0.0 
    influenced_grouped = []

    for i in range(num_groups):
        influenced_grouped.append(0.0)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
    results = pool.map(runIC_fair_deadline, [(G,S,num_groups,tau) for i in range(R)])
    pool.close()
    pool.join()

    for T, T_grouped in results: 
        
        influenced  += float(len(T)) / R

        for idx , T_g in enumerate(T_grouped): # This is list of lists 
        
            influenced_grouped[idx] += (len(T_g) / R)

    return (influenced, influenced_grouped)

def map_select_next_seed_greedy(inp):
    # selects greedily 
    G,S,v,num_groups,tau = inp
    R = 500
    priority = 0.0
    influences = []
    variance = 0.0
    if v not in S:
        for j in range(R): # run R times Random Cascade
        # for different objective change this priority selection 
            T, T_grouped = runIC_fair_deadline((G,S + [v], num_groups,tau))
            inf = len(T)
            #influences.append(float(inf))
            priority -= float(inf)/R

        #variance = np.sum((np.asarray(influences) + priority ) ** 2) / R # wrong time to compute variance 
    #input("Press Enter to continue...")

    return (v,priority)#,math.sqrt(variance))

def map_select_next_seed_lazy_greedy(inp):
    '''
    returns influence and list of different groups influenced
    '''
    G,S,num_groups,tau = inp
    #print(S)
    R = 500 #changing to 200 didn't help 
    influenced = 0.0 
    influenced_grouped = []

    for i in range(num_groups):
        influenced_grouped.append(0.0)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
    results = pool.map(runIC_fair_deadline, [(G,S,num_groups,tau) for i in range(R)])
    pool.close()
    pool.join()

    for T, T_grouped in results: 
        
        influenced  += float(len(T)) / R

        for idx , T_g in enumerate(T_grouped): # This is list of lists 
        
            influenced_grouped[idx] += (float(len(T_g)) / R)


    return (influenced, influenced, influenced_grouped)

def map_select_next_seed_lazy_greedy_pll(inp):
    '''
    returns influence and list of different groups influenced
    '''
    G,S,node,num_groups,tau = inp
    #print(S)
    R = 10000 # changing to 200 didn't help
    influenced = 0.0 
    influenced_grouped = []

    for i in range(num_groups):
        influenced_grouped.append(0.0)
    #pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
    vals = []
    # for i in range(R):
    #     vals.append(0)
    influence_interm = 0.0 
    for j in range(R): 
        T, T_grouped = runIC_fair_deadline((G,S + [node],num_groups,tau))
    #pool.close()
    #pool.join()

    #for T, T_grouped in results: 
        
        influenced  += (float(len(T)) / R)
        #influence_interm  += float(len(T))
        # if (j+1) % 500 == 0:
        #     vals.append(influence_interm / (j+1))

        for idx , T_g in enumerate(T_grouped): # This is list of lists 
        
            influenced_grouped[idx] += (float(len(T_g)) / R)

    # variance = 0.0
    # count = 0 
    # variance_list = []
    # for v in vals:
    #     variance += ((v - influenced)**2) # no no no no no no no this is the variance among different runs you have to calculate variance of influence with different values of R i.e. run this function 5 times with 500 then 1000 and so on and calculate variance 
    #     count += 1
    #     if count%500==0:
    #         variance_list.append(variance / count)
            #print( variance / count )

    return (influenced, influenced, influenced_grouped, node)#, vals)

def map_select_next_seed_log_lazy_greedy_pll(inp):
    '''
    returns influence and list of different groups influenced
    '''
    G,S,node,num_groups,tau = inp
    #print(S)
    R = 10000 #changing to 200 didn't help 
    influenced = 0.0 
    influenced_grouped = []
    proxy_infl = 0.0
    e = 1e-20
    for i in range(num_groups):
        influenced_grouped.append(0.0)

    for j in range(R): 
        T, T_grouped = runIC_fair_deadline((G,S + [node],num_groups,tau))
        influenced  += float(len(T)) / R
        
        for idx , T_g in enumerate(T_grouped): # This is list of lists 
            influenced_grouped[idx] += (float(len(T_g)) / R)

    for I_g in influenced_grouped:
        proxy_infl += math.log10(I_g + e)

    return (influenced, proxy_infl, influenced_grouped, node)

def map_select_next_seed_root_lazy_greedy_pll(inp):
    '''
    returns influence and list of different groups influenced
    '''
    G,S,node,gamma,num_groups,tau = inp
    #print(S)
    R = 10000 #changing to 200 didn't help 
    influenced = 0.0 
    influenced_grouped = []
    proxy_infl = 0.0

    for i in range(num_groups):
        influenced_grouped.append(0.0)

    for j in range(R): 
        T, T_grouped = runIC_fair_deadline((G,S + [node],num_groups,tau))
        influenced  += float(len(T)) / R

        for idx , T_g in enumerate(T_grouped): # This is list of lists 
            influenced_grouped[idx] += (float(len(T_g)) / R)

    for I_g in influenced_grouped:
        proxy_infl += I_g**(1.0/gamma)

    return (influenced, proxy_infl, influenced_grouped, node)

def map_select_next_seed_log_greedy_prev(inp):
    # selects greedily 
        G,S,v,gamma = inp
        R = 200
        priority = 0.0
        e = 1e-20
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair((G,S + [v]))
                priority -= (math.log10(float(len(T_a)) + 1e-20) + gamma * math.log10(float(len(T_b)) + 1e-20))/R

        return (v,priority)

def map_select_next_seed_log_greedy(inp):
    # selects greedily 
        G,S,v,gamma,num_groups,tau = inp
        R = 500 # 200 before 
        priority = 0.0
        e = 1e-20
        T_total_grouped = []
        for t in range(num_groups):
            T_total_grouped.append(0.0)
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_grouped = runIC_fair_deadline((G,S + [v],num_groups, tau))
                for t in range(num_groups):
                    T_total_grouped[t] += (float(len(T_grouped[t]))/R)
                
                #priority -= (math.log10(float(len(T_a)) + 1e-20) + gamma * math.log10(float(len(T_b)) + 1e-20))/R
            for t in T_total_grouped:
                priority -= math.log10(t + e) 

        return (v,priority)

def map_select_next_seed_log_lazy_greedy(inp):
    # selects greedily 
    G,S,gamma,num_groups,tau = inp
    
    e = 1e-20
    R = 500
    influenced = 0.0 
    influenced_grouped = []
    proxy_infl = 0.0 
    
    for i in range(num_groups):
        influenced_grouped.append(0.0)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
    results = pool.map(runIC_fair_deadline, [(G,S,num_groups,tau) for i in range(R)])
    pool.close()
    pool.join()
    for T, T_grouped in results: # results has results of all the R runs 
    
        influenced += float(len(T)) / R

        for idx , T_g in enumerate(T_grouped): # This is list of lists T_grouped has group wise influences 
        
            influenced_grouped[idx] += (float(len(T_g)) / R)

    for I_g in influenced_grouped:  

        proxy_infl += math.log10(I_g + e) 

    return (influenced, proxy_infl, influenced_grouped)

def map_select_next_seed_root_greedy(inp):
    # selects greedily 
    G,S,v,gamma,beta, num_groups, tau = inp
    R = 500
    priority = 0.0
    T_total_grouped =[]
    for t in range(num_groups):
        T_total_grouped.append(0.0)
    if v not in S:
        for j in range(R): # run R times Random Cascade
        # for different objective change this priority selection 
            T, T_grouped = runIC_fair_deadline((G,S + [v],num_groups, tau))

            for t in range(num_groups):
                T_total_grouped[t] += (float(len(T_grouped[t]))/R)
            

            #priority -= (float(len(T_a))**(1/gamma) + float(len(T_b))**(1/gamma))**beta/R
        #priority -= ((F_a)**(1.0/gamma) + (F_b)**(1.0/gamma))**beta
        for t in T_total_grouped:
            priority -= t**(1.0/gamma)  
    return (v,priority)

def map_select_next_seed_root_lazy_greedy(inp):
    # selects greedily 
    G,S,gamma,num_groups,tau = inp
    
    e = 1e-20
    R = 500
    influenced = 0.0 
    influenced_grouped = []
    proxy_infl = 0.0 
    
    for i in range(num_groups):
        influenced_grouped.append(0.0)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
    results = pool.map(runIC_fair_deadline, [(G,S,num_groups,tau) for i in range(R)])
    pool.close()
    pool.join()
    for T, T_grouped in results: # results has results of all the R runs 
    
        influenced += float(len(T)) / R

        for idx , T_g in enumerate(T_grouped): # This is list of lists T_grouped has group wise influences 
        
            influenced_grouped[idx] += (float(len(T_g)) / R)

    for I_g in influenced_grouped:  

        proxy_infl += I_g**(1.0/gamma) 

    return (influenced, proxy_infl, influenced_grouped)

def map_select_next_seed_root_majority_greedy(inp):
    # selects greedily 
        G,S,v,gamma = inp
        R = 100
        priority = 0.0
        F_a = 0.0 
        F_b = 0.0
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair_deadline((G,S + [v],num_groups, ))
                F_a += float(len(T_a))/ R
                F_b += float(len(T_b))/ R

                #priority -= (float(len(T_a))**(1/gamma) + float(len(T_b))**(1/gamma))**beta/R
            priority -= ((F_a)**(1.0/gamma)*0 + F_b)
        return (v,priority)

def map_select_next_seed_norm_greedy(inp):
    # selects greedily 
        G,S,v,gamma = inp
        R = 100
        priority = 0.0
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair((G,S))
                priority -= ((float(len(T_a))**(1/gamma) + float(len(T_b))**(1/gamma))**gamma)/R

        return (v,priority)

def map_select_next_seed_set_cover(inp):
    # selects greedily 
        G,S,v, num_groups, tau = inp
        R = 10000
        priority = 0.0
        priority_grouped = []
        for i in range(num_groups):
            priority_grouped.append(0.0)

        priority_a = 0.0
        priority_b = 0.0
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_grouped = runIC_fair_deadline((G, S + [v], num_groups, tau))
                priority += float(len(T))/R # not subratacting like other formulations adding a minus later 
                for i in range(num_groups):
                    priority_grouped[i] += float(len(T_grouped[i]))/R
                # priority_a += float(len(T_a))/R
                # priority_b += float(len(T_b))/R

        return (v,priority_grouped)

def generalGreedy_parallel_inf(G, k, p=.01):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    parallel computation of influence of the node, but, probably, since the computation is not that complex 
    '''
    #import time
    #start = time.time()
    #define map function
     #CC_parallel(G, seed_size, .01)

    #results = []#np.asarray([])
    R = 500 # number of times to run Random Cascade
    S = [] # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k):
        s = PQ() # priority queue

        for v in G.nodes():
            if v not in S:
                s.add_task(v, 0) # initialize spread value
                [priority, count, task] = s.entry_finder[v]
                pool = multiprocessing.Pool(multiprocessing.cpu_count()/2)
                results = pool.map(map_IC, [(G,S + [v],p)]*R)
                pool.close()
                pool.join() 
                s.add_task(v, priority - float(np.sum(results))/R)
                #for j in range(R): # run R times Random Cascade
                     #[priority, count, task] = s.entry_finder[v]
                  #  s.add_task(v, priority - float(len(runIC(G, S + [v], p)))/R) # add normalized spread value
        task, priority = s.pop_item()
        S.append(task)
        #print(i, k, time.time() - start)
    return S

def generalGreedy(G, k, p=.01):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    #import time
    #start = time.time()
    R = 200 # number of times to run Random Cascade
    S = [] # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k): # cannot parallellize 
        s = PQ() # priority queue

        for v in G.nodes(): 
            if v not in S:
                s.add_task(v, 0) # initialize spread value
                #[priority, count, task] = s.entry_finder[v]
                for j in range(R): # run R times Random Cascade The gain of parallelizing isn't a lot as the one runIC is not very complex maybe for huge graphs 
                    [priority, count, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(runIC(G, S + [v], p)))/R) # add normalized spread value

        task, priority = s.pop_item()
        print(task, priority)
        S.append(task)
        #print(i, k, time.time() - start)
    return S

def generalGreedy_node_parallel(filename, G, budget, tau , gamma = 1.0, beta = 1.0, type_algo = 1, num_groups = 2, wr = True):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: Influence grouped -- is a list of lists [[ I_g1(S1),I_g2(S1)....I_gn(S1)][ I_g_1(S1,S2) ..... ]
            seeds_grouped =   -- is list of lists of lists  [[[S_g1(among S1)],[S_g2( among S1)]....] S_g_1(among S1,S2) ..... ]
    '''   
    S = [] # set of selected nodes
    #S_hard = [2078,107,1912,1431,102,2384,1596,1322,1006,821,1211,1684,1833,3852,2351]
    influenced_grouped = [] # is a list of lists [[ I_g1(S1),I_g2(S1)....I_gn(S1)][ I_g_1(S1,S2) ..... ]
    seeds_grouped = [] # is list of lists of lists  [[[S_g1(among S1)],[S_g2( among S1)]....] S_g_1(among S1,S2) ..... ]
   
    seed_range = []
    if  type_algo == 1:
            filename = filename + f'_greedy_tau_{tau}_'

    elif type_algo == 2:
            filename = filename + f'_log_gamma_{gamma}_tau_{tau}_'

    elif type_algo == 3:
             filename = filename + f'_root_gamma_{gamma}_beta_{beta}_tau_{tau}_'

    elif type_algo == 4:
             filename = filename + f'_root_majority_gamma_{gamma}_beta_{beta}_tau_{tau}_'
   

    stats = ut.graph_stats(G, num_groups, print_stats = False)
    print(" in node thingy ")
    try :

        influenced_grouped, seeds_grouped = ut.read_files(filename + 'a')# change this to multiple groups
        # for I in influenced_grouped: 
        #     print(I)
            
        ut.write_paper_format(filename, influenced_grouped, seeds_grouped, stats[0])
        #print ( influenced_grouped )
        #print(seeds_grouped)
        for idx, (i , s) in enumerate(zip(influenced_grouped[-1], seeds_grouped[-1])):
            #print(f'influenced {i} group {idx}, seeds: {s} ')
            S += s
            # for seeds in s:
            #     print(f' seeds: {seeds}')
        
        if len(S) >= budget:
            #ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)
            # print(influenced_a)
            # print( "\n\n")
            # print(influenced_b)
            # print(" Seed length ", len(S))

            ut.plot_influence(influenced_grouped, seeds_grouped, filename, stats[0])

            return (influenced_grouped, seeds_grouped)
        else: 
            seed_range = range(budget - len(S))

    except FileNotFoundError:
        print( f'{filename} not Found ')
        
        seed_range = range(budget)

    #variance = []
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in seed_range: # cannot parallellize 

        influences = {}

        #for j in range(10):

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
         
        if type_algo == 1:
            results = pool.map(map_select_next_seed_greedy, ((G,S,v,num_groups,tau) for v in G.nodes()))
        elif type_algo == 2:
            results = pool.map(map_select_next_seed_log_greedy, ((G,S,v,gamma,num_groups,tau) for v in G.nodes()))
        elif type_algo == 3:
            results = pool.map(map_select_next_seed_root_greedy, ((G,S,v,gamma,beta,num_groups,tau) for v in G.nodes()))
        elif type_algo == 4:
            results = pool.map(map_select_next_seed_root_majority_greedy, ((G,S,v,gamma,num_groups,tau) for v in G.nodes()))


        pool.close()
        pool.join()

        s = PQ() # priority queue
        

        for v,priority in results: # run R times Random Cascade The gain of parallelizing isn't a lot as the one runIC is not very complex maybe for huge graphs
            if v in influences:
                influences[v].append(priority * -1)
            else:
                influences[v] = [priority * -1]
            s.add_task(v, priority)

        # input("Press Enter to continue...")
        # print(influences)
        # input("press enter to continue")

        node, priority = s.pop_item() # uncomment later 
        

        # #node = S_hard[i] # remove 

        S.append(node)

        Influenced, I_grouped = map_fair_IC((G,S, num_groups,tau)) # assuming it returns ordered lists of group influences 
        # #influenced_grouped.append(I_grouped)
      
        group = G.nodes[node]['group']

        print( f'{i+1} Selected Node is {node} from group, {group}, Infl: {Influenced}') 
        Infd = []
        for idx , I in enumerate(I_grouped): 
            Infd.append(I)
            print(f'group: {idx},  I = {I}')
        
        influenced_grouped.append(Infd)
        seeds_grouped.append(ut.order_seeds(G,S,num_groups)) # 
        # if j == 9:
        #     S.append(node)

        #input("Press Enter to continue...")
        # v = []
        # for key, value in influences.items():
        #     mean = np.sum(np.asarray(value)) / 10
        #     v.append(np.sum((np.asarray(value) - mean) ** 2)  / 10)
        # variance.append(np.max(np.asarray(v)** 0.5)) 
            #print(f'node, {key}, variance: {np.sum((np.asarray(value) - mean) ** 2)  / 50}')
        #input("Press Enter to continue...")

    # ut.write_files(filename, influenced_grouped, seeds_grouped)
    # #print("here")
    # ut.write_paper_format(filename, influenced_grouped, seeds_grouped,stats[0])
    # #print("std:", variance)
    # #print(np.sum(np.asarray(variance))/ len(variance))
    # ut.plot_influence(influenced_grouped, seeds_grouped, filename, stats[0])
    
    return (influenced_grouped, seeds_grouped) 

def generalGreedy_node_set_cover(filename, G, budget, tau, gamma_a = 1e-2, gamma_b = 0, type_algo = 1, num_groups = 2):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- fraction of population needs to be influenced in both groups 
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    #import time
    #start = time.time()
    #R = 200 # number of times to run Random Cascade
   
   
    stats = ut.graph_stats(G, num_groups, print_stats = False)

    if  type_algo == 1:
            filename = filename + f'_set_cover_reach_{budget}_tau_{tau}_no_groups_'
            budget = (budget/num_groups) # as we multiply the with num_groups 
    elif type_algo == 2:
            filename = filename + f'_set_cover_reach_{budget}_tau_{tau}_'
    elif type_algo == 3:
            filename = filename + f'_set_cover_timings_reach_{budget}_gamma_a_{gamma_a}_gamma_b_{gamma_b}_tau_{tau}_'
    elif type_algo == 4:
            filename = filename + f'_set_cover_timings_reach_{budget}_gamma_a_{gamma_a}_gamma_b_{gamma_a}_tau_{tau}_'

    reach = 0.0
    S = [] # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    influenced_grouped = [] # is a list of lists [[ I_g1(S1),I_g2(S1)....I_gn(S1)][ I_g_1(S1,S2) ..... ]
    seeds_grouped = [] # is list of lists of lists  [[[S_g1(among S1)],[S_g2( among S1)]....] S_g_1(among S1,S2) ..... ]

    try :

        influenced_grouped,seeds_grouped = ut.read_files(filename)

        print( f' Found file {filename} ******* \n \n ')
        ut.write_paper_format(filename, influenced_grouped, seeds_grouped,stats[0])
        # for now assume if you have a file it has all the results otherwise computer everything
        # reach = min(influenced_a[-1]/stats['group_a'],budget) + min(influenced_b[-1]/stats['group_b'],budget)

        # S = seeds_a[-1] + seeds_b[-1]
        # if reach >= 2* budget:
        #     #ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)
        #     print(influenced_a)
        #     print( "\n\n")
        #     print(influenced_b)            
        #     print(f" reach: {reach}")
        #     ut.plot_influence(influenced_a, influenced_b, len(S), filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
        #     return (influenced, influenced_a, influenced_b, seeds_a, seeds_b)
        return (influenced_grouped, seeds_grouped)
        
    except FileNotFoundError:
        print( f'{filename} not Found ')
        
    i = 0
    while reach < num_groups * budget: # cannot parallellize 

        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)

        if type_algo == 1:
            results = pool.map(map_select_next_seed_set_cover, ((G,S,v,num_groups,tau) for v in G.nodes()))
        elif  type_algo == 2:
            results = pool.map(map_select_next_seed_set_cover, ((G,S,v,num_groups,tau) for v in G.nodes()))
        elif type_algo == 3:
            results = pool.map(map_IC_timing, ((G,S,v,gamma_a, gamma_b) for v in G.nodes()))
        elif type_algo == 4:
            results = pool.map(map_IC_timing, ((G,S,v,gamma_a, gamma_a) for v in G.nodes()))

        pool.close()
        pool.join()

        s = PQ() # priority queue
        for v,priority_grouped in results:# 

            p_to_add = 0.0
            if type_algo == 1:
                p_to_add = np.sum(np.asarray(priority_grouped)) / np.sum(np.asarray(stats[0])) #We are just adding total fraction population influenced i.e. just greedy
            else:
                for idx, p in enumerate(priority_grouped):
                    p_to_add += min(p / stats[0][idx], budget) # stats[0] are group populations

            s.add_task(v, -1 * p_to_add)

        node, priority = s.pop_item()
        #priority = -priority # as the current priority is negative fraction 
        S.append(node)

        Influenced, I_grouped = map_fair_IC((G,S, num_groups,tau)) # assuming it returns ordered lists of group influences 
        # #influenced_grouped.append(I_grouped)
      
        group = G.nodes[node]['group']

        print( f'{i+1} Selected Node is {node} from group, {group}') 
        Infd = []
        for idx , I in enumerate(I_grouped): 
            Infd.append(I)
            print(f'group: {idx},  I = {I/stats[0][idx]}')
        
        influenced_grouped.append(Infd)
        seeds_grouped.append(ut.order_seeds(G,S,num_groups)) # 

        print("total influence: " , np.sum(np.asarray(Infd)) / np.sum(np.asarray(stats[0])))
        #print(S)
        reach = (-1 * priority)
        i+=1

    ut.write_files(filename, influenced_grouped, seeds_grouped)
    #print("here")
    ut.write_paper_format(filename, influenced_grouped, seeds_grouped,stats[0])
    #print("std:", variance)
    #print(np.sum(np.asarray(variance))/ len(variance))
    ut.plot_influence(influenced_grouped, seeds_grouped, filename, stats[0])
    
    

    return (influenced_grouped, seeds_grouped)

def lazyGreedy_node_set_cover(filename, G, budget, tau, gamma_a = 1e-2, gamma_b = 0, type_algo = 1, num_groups = 2, budget_range = [], batch_size = 60, fact = 2):


    results_exist, filename, budget, influenced_grouped, seeds_grouped = ut.get_results_cover(filename, G, budget, tau, type_algo, num_groups)
    
    if results_exist:
        return (influenced_grouped, seeds_grouped)

    stats = ut.graph_stats(G, num_groups, print_stats = False)

    reach = 0.0
    S = [] # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    influenced_grouped = [] # is a list of lists [[ I_g1(S1),I_g2(S1)....I_gn(S1)][ I_g_1(S1,S2) ..... ]
    seeds_grouped = [] # is list of lists of lists  [[[S_g1(among S1)],[S_g2( among S1)]....] S_g_1(among S1,S2) ..... ]
    batch_influence = {}

    def priority_calculator(type_algo, priority_grouped, print_ = False):
        p_to_add = 0.0
        if type_algo == 1:
            p_to_add = np.sum(np.asarray(priority_grouped)) / np.sum(np.asarray(stats[0])) #We are just adding total fraction population influenced i.e. just greedy    
        else:
            for idx, p in enumerate(priority_grouped):
                p_to_add += min(p / stats[0][idx], budget) # stats[0] are group populations
                if print_:
                    print(f'G: {idx} : infl {min(p / stats[0][idx], budget)}')
        return p_to_add

    def print_results(I_grouped,i):

        group = G.nodes[node]['group']

        print( f'{i+1} Selected Node is {node} from group, {group}') 
        Infd = []
        for idx , I in enumerate(I_grouped): 
            Infd.append(I)
            print(f'group: {idx},  I = {I/stats[0][idx]}')

        print("total influence: " , np.sum(np.asarray(Infd)) / np.sum(np.asarray(stats[0])))

        return (Infd, ut.order_seeds(G,S,num_groups))
        #print(S)

    # Calculating first influences 
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    
    nodes_set = G.nodes() #ut.get_seed_sample(G.nodes(), 5000, G, filename)
    print(len(nodes_set))
    #print( nodes_set ) 
    #input(' Enter a key')
    results = pool.map(map_select_next_seed_set_cover, ((G,S,v,num_groups,tau) for v in nodes_set))

    pool.close()
    pool.join()
    s = PQ() # priority queue

    
    #def queue_entry(results, type_algo)
    for v,priority_grouped in results: 
        batch_influence[v] =  priority_grouped
        p_to_add = priority_calculator(type_algo, priority_grouped)
        s.add_task(v, -1 * p_to_add)

    node, priority = s.pop_item()  
    reach = (-1 * priority)
    S.append(node)
    #results = map_select_next_seed_lazy_greedy((G,S,num_groups,tau)) # change this 
    #_, _, priority_grouped = results

    Infd, ordered_seeds = print_results(batch_influence[node],0)
    influenced_grouped.append(Infd)
    seeds_grouped.append(ordered_seeds)#

    #prev_spread = -1 * priority

    print('frist iteration of the nodes and their influences updated ... ')


    i = 1
    batch_influence = {}
    nodes = []
    j = 0
    nodes = ut.get_batch(s, batch_size)
    # print(f'num nodes: {len(nodes)}, {len(nodes[-1])} {num_groups}  {budget}')
    # input()
    count = 0
    while reach < num_groups * budget: # cannot parallellize 

        while(True):

            node, priority_a = s.pop_item() # pop the top node

            if node not in batch_influence:
                print( f'Node not found now computing next batch {(j+1) * len(nodes[j])}')
                pool = multiprocessing.Pool(int(multiprocessing.cpu_count() * fact))
                results = pool.map(map_select_next_seed_lazy_greedy_pll,((G,S,n,num_groups,tau) for n in nodes[j]))
                pool.close()
                pool.join()
                for Influenced_r, proxy_infl_r, I_grouped_r, n_r in results:
                    batch_influence[n_r] = (Influenced_r, proxy_infl_r, I_grouped_r)
                print(f" Next batch computed!") 
                j+=1

            #results = map_select_next_seed_lazy_greedy((G,S+[node],num_groups,tau)) # revaluate

            Influenced, proxy_infl, I_grouped = batch_influence[node] 

            new_influence = priority_calculator(type_algo, I_grouped)

            gain = new_influence - reach

            s.add_task(node, -1*gain) 

            node_, priority = s.pop_item() # if it retains its position
            count += 1
            # if count >= j * len(nodes[j]):
            #     print( f' *** cout is:{count} first top: {node}, priority {priority_a}, updated to {-1 * gain} new top: {node_} priority: {priority} length new nodes: {len(batch_influence['key'])}, grouped infl {I_grouped} ')
            #     input("Press Enter to continue...")
            if priority == priority_a:
                if node_ in batch_influence: # if it has been calculate with new seed set 
                    n_Influenced, n_proxy_infl, n_I_grouped = batch_influence[node_]
                    n_prirority = priority_calculator(type_algo, n_I_grouped)
                    n_prirority = -1* n_prirority
                    if n_prirority == priority:
                    #node, priority_a = s.pop_item()
                        s.remove(node)
                        s.add_task(node_, priority)
                        S.append(node) # then add to seed node 
                        reach = new_influence
                        #prev_spread = new_influence
                        print( f'influence of the node : {-1 * priority}')
                        print( f' Number of total evaluations {count}')
                        i+=1
                        batch_influence = {}
                        nodes = ut.get_batch(s, batch_size)
                        j = 0
                        count = 0
                        break

            elif node == node_:
                S.append(node) # then add to seed node 
                reach = new_influence
                #prev_spread = new_influence
                print( f'influence of the node : {-1 * priority}')
                print( f' Number of total evaluations {count}')
                i+=1
                batch_influence = {}
                nodes = ut.get_batch(s, batch_size)
                j = 0
                count = 0
                break
            else:
                # otherwise keep looking
                s.add_task(node_, priority)

        print(f'********* reach {reach} *******\n \n \n')
        # This will act as the test , as we have already selected the seeds now we are running it again and present those statistics 
        #Influenced, I_grouped = map_fair_IC((G,S, num_groups,tau)) # taking too long 
        Infd, ordered_seeds = print_results(I_grouped,i-1)
        influenced_grouped.append(Infd)
        seeds_grouped.append(ordered_seeds)
        
        
    # The last write       
    ut.write_files(filename, influenced_grouped, seeds_grouped)
    #print("here")
    ut.write_paper_format(filename, influenced_grouped, seeds_grouped, stats[0])
    #print("std:", variance)
    #print(np.sum(np.asarray(variance))/ len(variance))
    ut.plot_influence(influenced_grouped, seeds_grouped, filename, stats[0])

    return (influenced_grouped, seeds_grouped)
    
def lazyGreedy_node_parallel(filename, G, budget, tau , gamma = 1.0, beta = 1.0, type_algo = 1, num_groups = 2, batch_size = 60, fact = 1):

    results_exist, filename, seed_range, S, influenced_grouped, seeds_grouped = ut.get_results_budget(filename, budget, tau, gamma, beta, type_algo, G, num_groups)
    if results_exist:
       return (influenced_grouped, seeds_grouped) 
    
    S = []
    influenced_grouped = []
    seeds_grouped = []
    stats = ut.graph_stats(G, num_groups, print_stats = False)

    print( f'\n\n batch size is: {batch_size} groups: {num_groups}')


    #
    def find_one_seed():

        influences = {}
        batch_influence = {}
        #for j in range(10):
        #nodes_set = ut.get_seed_sample(G.nodes(), 5000, G, filename) # for instagram dataset 
        nodes_set = G.nodes() #ut.get_seed_sample(G.nodes(), 5000, G, filename)

        print( f'seed candidates are {len(nodes_set)}')
        # s = PQ() 
        # for v in nodes_set:
        #    Influenced_r, proxy_infl_r, I_grouped_r =  map_select_next_seed_lazy_greedy((G,S+[v],num_groups,tau))
        #    s.add_task(n_r, -1 * proxy_infl_r)
        #    batch_influence[v] = (Influenced_r, proxy_infl_r, I_grouped_r)
        # Here I am just doing multi processing over all the nodes.
        #values_list = {} # influence values over multiple runs for each node 
        runs = 1#0 
        for i in range(runs):
            pool = multiprocessing.Pool(int(multiprocessing.cpu_count() -1 )) #*fact))
            
            if type_algo == 1:
                results = pool.map(map_select_next_seed_lazy_greedy_pll, ((G,S,v,num_groups,tau) for v in nodes_set))
            elif type_algo == 2:
                results = pool.map(map_select_next_seed_log_lazy_greedy_pll, ((G,S,v,num_groups,tau) for v in nodes_set))
            elif type_algo == 3:
                results = pool.map(map_select_next_seed_root_lazy_greedy_pll, ((G,S,v,gamma,num_groups,tau) for v in nodes_set))
            
            #print(res.get(timeout=1))
            pool.close()
            pool.join()

            s = PQ() 
            for Influenced_r, proxy_infl_r, I_grouped_r, n_r in results:

                #print( f' node: {n_r}, infl: {vals} ')
                # for idx, v in enumerate(variance_list):
                #     print( idx, v )

                #input( ' Press a key to proceed... ')
                # if n_r in values_list:
                #     values_list[n_r].append(vals)
                # else:
                #     values_list[n_r] = [vals] # appending lists 

                s.add_task(n_r, -1 * proxy_infl_r)
                batch_influence[n_r] = (Influenced_r, proxy_infl_r, I_grouped_r)
            #print(f' Run {i+1} is done ')

        # average_nodes = []
        # variance_nodes = []
        # #print(variance_list)
        # for key in values_list:
        #     #variance_list[key] contains [[influ_500, influ_1000, .... influ_20000]_run_1 , 
        #     #[influ_500, influ_1000, .... influ_20000]_run_2 ... ]  
        #     average = np.zeros(len(values_list[key][0])) #assuming for other runs the length of the variances list will stay the same 
        #     for elem in values_list[key]:
        #         average = np.add(average, elem)

        #     average = average / runs # or maybe add ? 
            
        #     interm_list = []
        #     for elem in values_list[key]:
                
        #         interm_list.append(np.power((np.subtract(elem , average)), 2))

        #     variances = np.zeros(len(interm_list[0]))
        #     for elem in interm_list:
        #         variances = np.add(variances , elem)
        #         #print(elem)
        #     variances = variances/ runs
        #     average_nodes.append(average.tolist())
        #     variance_nodes.append(variances.tolist())
        #     #print(f' node: {key} average: {average} variance: {variances}') # this should print variances of 10 runs for each node for 500, 1000, ... so on no. of monte carlo samples 
        # print(f' average: {average_nodes} variance: {variance_nodes}')
        # ut.write_averages_variance(average_nodes, variance_nodes, filename)
        return s, batch_influence 

    
    
    s, batch_influence = find_one_seed()
    #input('Stopping here .... ')
    node, priority = s.pop_item()
    print(f'node, removed : {node}')
    S.append(node)
    #Influenced, I_grouped = map_fair_IC((G,S, num_groups,tau))
    Influenced, proxy_infl, I_grouped = batch_influence[node]
    group = G.nodes[node]['group']
    print( f'{1} Selected Node is {node} from group, {group}, Infl: {Influenced}') 

     

    Infd = []
    for idx , I in enumerate(I_grouped):
        Infd.append(I)
        print(f'group: {idx},  I = {I}')

    influenced_grouped.append(Infd)
    seeds_grouped.append(ut.order_seeds(G,S,num_groups))
    #
    #first seed selected already 
    
 
    seed_range = range(budget-1)
    prev_spread = -1 * priority 
    
    batch_influence = {}
    nodes = []
    j = 0
    nodes = ut.get_batch(s, batch_size) # returns list of list of batches to compute 
   
    print( f'num of batches {len(nodes)} ')
    count = 0

    for i in seed_range:
        print( f' \n\n\n Searching seed no. {i+2} \n \n \n ')
        while(True):
            node, priority = s.pop_item()
            if node not in batch_influence: 
                pool = multiprocessing.Pool(multiprocessing.cpu_count() -1 ) #* fact)
                if type_algo == 1:
                    results = pool.map(map_select_next_seed_lazy_greedy_pll,((G,S,n,num_groups,tau) for n in nodes[j]))    
                elif type_algo == 2:
                    results = pool.map(map_select_next_seed_log_lazy_greedy_pll,((G,S,n,num_groups,tau) for n in nodes[j]))
                elif type_algo == 3:
                    results = pool.map(map_select_next_seed_root_lazy_greedy_pll,((G,S,n,gamma,num_groups,tau) for n in nodes[j]))
                
                pool.close()
                pool.join()
            
                for Influenced_r, proxy_infl_r, I_grouped_r, n_r in results:
                    # for idx, v in enumerate(variance_list):
                    #     print( idx, v )
                    batch_influence[n_r] = (Influenced_r, proxy_infl_r, I_grouped_r)
                print(f" Node not found size: {(j+1) * len(nodes[j])}")
                j+=1
               
            Influenced, proxy_infl, I_grouped = batch_influence[node]
            
            gain = proxy_infl - prev_spread 

            s.add_task(node, -1*gain)


            node_, priority = s.pop_item()

            if node == node_:
                S.append(node)
                prev_spread = proxy_infl
                batch_influence = {}
                nodes = ut.get_batch(s, batch_size)
                j = 0
                break
            else:
                s.add_task(node_, priority)
                count+=1
                   
        group = G.nodes[node]['group']
        print( f'{i+2} Selected Node is {node} from group, {group}, Infl: {Influenced}')
        #Influenced, I_grouped = map_fair_IC((G,S, num_groups,tau))
        Infd = []
        for idx , I in enumerate(I_grouped): 
            Infd.append(I)
            print(f'group: {idx},  I = {I}')
        
        influenced_grouped.append(Infd)
        seeds_grouped.append(ut.order_seeds(G,S,num_groups))
        
    ut.write_files(filename, influenced_grouped, seeds_grouped)
    # #print("here")
    ut.write_paper_format(filename, influenced_grouped, seeds_grouped,stats[0])
    # #print("std:", variance)
     #print(np.sum(np.asarray(variance))/ len(variance))
    ut.plot_influence(influenced_grouped, seeds_grouped, filename, stats[0])

    return (influenced_grouped, seeds_grouped)

class reduced_lazy_greedy():

    
    def __init__(self,G):
        import sys 
        #from multiprocessing import shared_memory
        manager = multiprocessing.Manager()
        print( len( G.edges()), len(G.nodes()))
        print(" memory:", sys.getsizeof(G.edges()) + sys.getsizeof(G.nodes()))

        self.graphs = manager.list(ut.generate_graphs(G,10000))

        # temp = ut.generate_graphs(G,5)
        # temp = np.asarray(temp)
        # shm = shared_memory.SharedMemory(create=True, size=temp.nbytes)
        # self.graphs = np.ndarray(temp.shape, dtype=temp.dtype, buffer=shm.buf)
        # self.graphs[:] = temp[:]
        # del(temp)
        #print(self.graphs[0])
        print(f'number of graphs is: {len(self.graphs)}')
        input('...')

    def fib(n):
        if n == 0 or n ==1 :
            return 1
        else:
            return fib(n-1) + fib(n-2)

    def map_select_next_seed_lazy_greedy_pll_pre_built_graphs(self, inp):
        '''
        returns influence and list of different groups influenced
        '''
        S,node,num_groups,tau = inp

        R = len(self.graphs)
        
        influenced = 0.0 
        influenced_grouped = []

        for i in range(num_groups):
            influenced_grouped.append(0.0)
        
        influence_interm = 0.0 

        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)

        results = pool.map(runIC_fair_deadline_reduced,((self.graphs[j],S + [node],num_groups,tau) for j in range(R)))

        pool.close()
        pool.join()

        for j in range(R): 
            # if node not in self.graphs[j].nodes():
            #     for s in S:

            #     T = S + [node]
            #     #continue
            # else:
            #T, T_grouped = runIC_fair_deadline_reduced((self.graphs[j],S + [node],num_groups,tau))
            T,T_grouped = results[j]
              
            influenced  += (float(len(T)) / R)  

            for idx, T_g in enumerate(T_grouped): # This is list of lists 
            
                influenced_grouped[idx] += (float(len(T_g)) / R)

        return (influenced, influenced, influenced_grouped, node)#, vals)

    def map_select_next_seed_lazy_greedy_log_pll_pre_built_graphs(self, inp):
        '''
        returns influence and list of different groups influenced
        '''
        S,node,num_groups,tau = inp

        R = len(self.graphs)
        
        influenced = 0.0 
        influenced_grouped = []
        proxy_infl = 0.0 

        e = 1e-20

        for i in range(num_groups):
            influenced_grouped.append(0.0)
        
        influence_interm = 0.0 

        # pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)

        # results = pool.map(runIC_fair_deadline_reduced,((self.graphs[j],S + [node],num_groups,tau) for j in range(R)))
        # pool.close()
        # pool.join()

        for j in range(R):
            
            T, T_grouped = runIC_fair_deadline_reduced((self.graphs[j],S + [node],num_groups,tau))
            #T, T_grouped = results [j]
            
            # while True:
            #     pass

            influenced  += (float(len(T)) / R)

            for idx, T_g in enumerate(T_grouped): # This is list of lists 
            
                influenced_grouped[idx] += (float(len(T_g)) / R)# - fib(j))

        for I_g in influenced_grouped:
            proxy_infl += math.log10(I_g + e)

        return (influenced, proxy_infl, influenced_grouped, node)

    def map_select_next_seed_lazy_greedy_root_pll_pre_built_graphs(self, inp):
        '''
        returns influence and list of different groups influenced
        '''
        S,node,num_groups,tau,gamma = inp

        R = len(self.graphs)
        
        influenced = 0.0 
        influenced_grouped = []
        proxy_infl = 0.0 

        for i in range(num_groups):
            influenced_grouped.append(0.0)
        
        influence_interm = 0.0 

        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)

        results = pool.map(runIC_fair_deadline_reduced,((self.graphs[j],S + [node],num_groups,tau) for j in range(R)))
        pool.close()
        pool.join()

        for j in range(R): 

            # if node not in self.graphs[j].nodes():
            #     for s in S:

            #     T = S + [node]
            #     #continue
            # else:
            #T, T_grouped = runIC_fair_deadline_reduced((self.graphs[j],S + [node],num_groups,tau))
            T, T_grouped = results[j]
              
            influenced  += (float(len(T))/R)

            for idx, T_g in enumerate(T_grouped): # This is list of lists 
                influenced_grouped[idx] += (float(len(T_g)) / R)


        for I_g in influenced_grouped:
            proxy_infl += I_g**(1.0/gamma)

        return (influenced, proxy_infl, influenced_grouped, node)

    def lazyGreedy_node_parallel_pre_built_graphs(self, filename, G, budget, tau , gamma = 1.0, beta = 1.0, type_algo = 1, num_groups = 2, batch_size = 60, fact = 1):

        results_exist, filename, seed_range, S, influenced_grouped, seeds_grouped = ut.get_results_budget(filename, budget, tau, gamma, beta, type_algo, G, num_groups)
        if results_exist:
           return (influenced_grouped, seeds_grouped) 
        
        S = []
        influenced_grouped = []
        seeds_grouped = []
        stats = ut.graph_stats(G, num_groups, print_stats = False)

        print( f'\n\n batch size is: {batch_size} groups: {num_groups}')

        #
        def find_one_seed():

            influences = {}
            batch_influence = {}
            #for j in range(10):
            nodes_set = ut.get_seed_sample(G.nodes(), 5000, G, filename)
            print( f'seed candidates are {len(nodes_set)}')
            
            # Here I am just doing multi processing over all the nodes.
            values_list = {} # influence values over multiple runs for each node 
            runs = 1#0 
            for i in range(runs):
                print('running log lazy reduced the first run parallel for nodes  ')
                #print('running log lazy reduced the first run parallel for MC')
                pool = multiprocessing.Pool(int(multiprocessing.cpu_count() -1 )) #*fact))
                
                if type_algo == 1:
                    results = pool.map(self.map_select_next_seed_lazy_greedy_pll_pre_built_graphs, ((S,v,num_groups,tau) for v in nodes_set))
                elif type_algo == 2: 
                    
                    
                    results = pool.map(self.map_select_next_seed_lazy_greedy_log_pll_pre_built_graphs, ((S,v,num_groups,tau) for v in nodes_set))
                    # print('running log lazy reduced the first run parallel for MC')
                     # results =[]
                     # for v in nodes_set:
                     #     results.append(self.map_select_next_seed_lazy_greedy_log_pll_pre_built_graphs((S,v,num_groups,tau)))

                elif type_algo == 3:
                    #results = pool.map(self.map_select_next_seed_lazy_greedy_root_pll_pre_built_graphs, ((S,v,num_groups,tau, gamma) for v in nodes_set))

                    results =[]
                    for v in nodes_set:
                        results.append(self.map_select_next_seed_lazy_greedy_log_pll_pre_built_graphs((S,v,num_groups,tau)))
                
                #print(res.get(timeout=1))
                pool.close()
                pool.join()

                s = PQ() 
                for Influenced_r, proxy_infl_r, I_grouped_r, n_r in results:

                   

                    s.add_task(n_r, -1 * proxy_infl_r)
                    batch_influence[n_r] = (Influenced_r, proxy_infl_r, I_grouped_r)
                
            return s, batch_influence 

        
        
        s, batch_influence = find_one_seed()
        #input('Stopping here .... ')
        node, priority = s.pop_item()
        print(f'node, removed : {node}')
        S.append(node)
        #Influenced, I_grouped = map_fair_IC((G,S, num_groups,tau))
        Influenced, proxy_infl, I_grouped = batch_influence[node]
        group = G.nodes[node]['group']
        print( f'{1} Selected Node is {node} from group, {group}, Infl: {Influenced}') 

         

        Infd = []
        for idx , I in enumerate(I_grouped):
            Infd.append(I)
            print(f'group: {idx},  I = {I}')

        influenced_grouped.append(Infd)
        seeds_grouped.append(ut.order_seeds(G,S,num_groups))
        #
        #first seed selected already 
        
     
        seed_range = range(budget-1)
        prev_spread = -1 * priority 
        
        batch_influence = {}
        nodes = []
        j = 0
        nodes = ut.get_batch(s, batch_size) # returns list of list of batches to compute 
       
        print( f'num of batches {len(nodes)} ')
        count = 0

        for i in seed_range:
            print( f' \n\n\n Searching seed no. {i+2} \n \n \n ')
            while(True):
                node, priority = s.pop_item()
                if node not in batch_influence: 
                    #pool = multiprocessing.Pool(multiprocessing.cpu_count() -1 ) #* fact)
                    results = []
                    if type_algo == 1:
                        results = pool.map(self.map_select_next_seed_lazy_greedy_pll_pre_built_graphs,((graphs,S,n,num_groups,tau) for n in nodes[j]))    
                    elif type_algo == 2:

                        results.append(map_select_next_seed_lazy_greedy_log_pll_pre_built_graphs(S,n,num_groups,tau))
                        #results = pool.map(map_select_next_seed_log_lazy_greedy_pll,((G,S,n,num_groups,tau) for n in nodes[j]))
                    elif type_algo == 3:
                        results.append(map_select_next_seed_lazy_greedy_root_pll_pre_built_graphs(S,n,num_groups,tau,gamma))
                        #results = pool.map(map_select_next_seed_root_lazy_greedy_pll,((G,S,n,gamma,num_groups,tau) for n in nodes[j]))
                    
                    #pool.close()
                    #pool.join()
                
                    for Influenced_r, proxy_infl_r, I_grouped_r, n_r in results:
                        # for idx, v in enumerate(variance_list):
                        #     print( idx, v )
                        batch_influence[n_r] = (Influenced_r, proxy_infl_r, I_grouped_r)
                    print(f" Node not found size: {(j+1) * len(nodes[j])}")
                    j+=1
                   
                Influenced, proxy_infl, I_grouped = batch_influence[node]
                
                gain = proxy_infl - prev_spread 

                s.add_task(node, -1*gain)


                node_, priority = s.pop_item()

                if node == node_:
                    S.append(node)
                    prev_spread = proxy_infl
                    batch_influence = {}
                    nodes = ut.get_batch(s, batch_size)
                    j = 0
                    break
                else:
                    s.add_task(node_, priority)
                    count+=1
                       
            group = G.nodes[node]['group']
            print( f'{i+2} Selected Node is {node} from group, {group}, Infl: {Influenced}')
            #Influenced, I_grouped = map_fair_IC((G,S, num_groups,tau))
            Infd = []
            for idx , I in enumerate(I_grouped): 
                Infd.append(I)
                print(f'group: {idx},  I = {I}')
            
            influenced_grouped.append(Infd)
            seeds_grouped.append(ut.order_seeds(G,S,num_groups))
            
        ut.write_files(filename, influenced_grouped, seeds_grouped)
        # #print("here")
        ut.write_paper_format(filename, influenced_grouped, seeds_grouped,stats[0])
        # #print("std:", variance)
         #print(np.sum(np.asarray(variance))/ len(variance))
        ut.plot_influence(influenced_grouped, seeds_grouped, filename, stats[0])

        return (influenced_grouped, seeds_grouped)

    def lazyGreedy_node_set_cover_pre_built_graphs(self, filename, G, budget, tau, gamma_a = 1e-2, gamma_b = 0, type_algo = 1, num_groups = 2, budget_range = [], batch_size = 60, fact = 2):

        results_exist, filename, budget, influenced_grouped, seeds_grouped = ut.get_results_cover(filename, G, budget, tau, type_algo, num_groups)

        if results_exist:
            return (influenced_grouped, seeds_grouped)

        stats = ut.graph_stats(G, num_groups, print_stats = False)

        reach = 0.0
        S = [] # set of selected nodes
        # add node to S if achieves maximum propagation for current chosen + this node
        influenced_grouped = [] # is a list of lists [[ I_g1(S1),I_g2(S1)....I_gn(S1)][ I_g_1(S1,S2) ..... ]
        seeds_grouped = [] # is list of lists of lists  [[[S_g1(among S1)],[S_g2( among S1)]....] S_g_1(among S1,S2) ..... ]
        batch_influence = {}

        def priority_calculator(type_algo, priority_grouped, print_ = False):
            p_to_add = 0.0
            if type_algo == 1:
                p_to_add = np.sum(np.asarray(priority_grouped)) / np.sum(np.asarray(stats[0])) #We are just adding total fraction population influenced i.e. just greedy    
            else:
                for idx, p in enumerate(priority_grouped):
                    p_to_add += min(p / stats[0][idx], budget) # stats[0] are group populations
                    if print_:
                        print(f'G: {idx} : infl {min(p / stats[0][idx], budget)}')
            return p_to_add

        def print_results(I_grouped,i):

            group = G.nodes[node]['group']

            print( f'{i+1} Selected Node is {node} from group, {group}') 
            Infd = []
            for idx , I in enumerate(I_grouped): 
                Infd.append(I)
                print(f'group: {idx},  I = {I/stats[0][idx]}')

            print("total influence: " , np.sum(np.asarray(Infd)) / np.sum(np.asarray(stats[0])))

            return (Infd, ut.order_seeds(G,S,num_groups))
            #print(S)

        # Calculating first influences 
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        
        nodes_set = ut.get_seed_sample(G.nodes(), 5000, G, filename)
        #print( nodes_set ) 
        #input(' Enter a key')
        
        for v in nodes_set:
            self.map_select_next_seed_lazy_greedy_pll_pre_built_graphs(S,v,num_groups,tau)

        #results = pool.map(map_select_next_seed_set_cover, ((G,S,v,num_groups,tau) for v in nodes_set)) 
        #results = pool.map(self.map_select_next_seed_lazy_greedy_pll_pre_built_graphs, ((S,v,num_groups,tau) for v in nodes_set))

        #pool.close()
        #pool.join()
        s = PQ() # priority queue

        
        #def queue_entry(results, type_algo)
        for _,_, priority_grouped,v in results: 
            batch_influence[v] =  priority_grouped
            p_to_add = priority_calculator(type_algo, priority_grouped)
            s.add_task(v, -1 * p_to_add)

        node, priority = s.pop_item()  
        reach = (-1 * priority)
        S.append(node)
        #results = map_select_next_seed_lazy_greedy((G,S,num_groups,tau)) # change this 
        #_, _, priority_grouped = results

        Infd, ordered_seeds = print_results(batch_influence[node],0)
        influenced_grouped.append(Infd)
        seeds_grouped.append(ordered_seeds)#

        #prev_spread = -1 * priority

        print('frist iteration of the nodes and their influences updated ... ')

        i = 1
        batch_influence = {}
        nodes = []
        j = 0
        nodes = ut.get_batch(s, batch_size)
        count = 0

        while reach < num_groups * budget: # cannot parallellize 

            while(True):

                node, priority_a = s.pop_item() # pop the top node

                if node not in batch_influence:
                    print( f'Node not found now computing next batch {(j+1) * len(nodes[j])}')
                    pool = multiprocessing.Pool(int(multiprocessing.cpu_count() -1 )) #* fact))
                    results = pool.map(self.map_select_next_seed_lazy_greedy_pll_pre_built_graphs,((S,n,num_groups,tau) for n in nodes[j]))
                    pool.close()
                    pool.join()
                    for Influenced_r, proxy_infl_r, I_grouped_r, n_r in results:
                        batch_influence[n_r] = (Influenced_r, proxy_infl_r, I_grouped_r)
                    print(f" Next batch computed!") 
                    j+=1

                #results = map_select_next_seed_lazy_greedy((G,S+[node],num_groups,tau)) # revaluate

                Influenced, proxy_infl, I_grouped = batch_influence[node] 

                new_influence = priority_calculator(type_algo, I_grouped)

                gain = new_influence - reach

                s.add_task(node, -1*gain) 

                node_, priority = s.pop_item() # if it retains its position
                count += 1
                
                if priority == priority_a:
                    if node_ in batch_influence: # if it has been calculate with new seed set 
                        n_Influenced, n_proxy_infl, n_I_grouped = batch_influence[node_]
                        n_prirority = priority_calculator(type_algo, n_I_grouped)
                        n_prirority = -1 * n_prirority
                        if n_prirority == priority:
                        #node, priority_a = s.pop_item()
                            s.remove(node)
                            s.add_task(node_, priority)
                            S.append(node) # then add to seed node 
                            reach = new_influence
                            #prev_spread = new_influence
                            print( f'influence of the node : {-1 * priority}')
                            print( f' Number of total evaluations {count}')
                            i+=1
                            batch_influence = {}
                            nodes = ut.get_batch(s, batch_size)
                            j = 0
                            count = 0
                            break

                elif node == node_:
                    S.append(node) # then add to seed node 
                    reach = new_influence
                    #prev_spread = new_influence
                    print( f'influence of the node : {-1 * priority}')
                    print( f' Number of total evaluations {count}')
                    i+=1
                    batch_influence = {}
                    nodes = ut.get_batch(s, batch_size)
                    j = 0
                    count = 0
                    break
                else:
                    # otherwise keep looking
                    s.add_task(node_, priority)

            print(f'********* reach {reach} *******\n \n \n')
            # This will act as the test , as we have already selected the seeds now we are running it again and present those statistics 
            #Influenced, I_grouped = map_fair_IC((G,S, num_groups,tau)) # taking too long 
            Infd, ordered_seeds = print_results(I_grouped,i-1)
            influenced_grouped.append(Infd)
            seeds_grouped.append(ordered_seeds)
            
            
        # The last write       
        ut.write_files(filename, influenced_grouped, seeds_grouped)
        #print("here")
        ut.write_paper_format(filename, influenced_grouped, seeds_grouped, stats[0])
        #print("std:", variance)
        #print(np.sum(np.asarray(variance))/ len(variance))
        ut.plot_influence(influenced_grouped, seeds_grouped, filename, stats[0])

        return (influenced_grouped, seeds_grouped)


