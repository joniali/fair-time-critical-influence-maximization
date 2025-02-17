import numpy as np
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from operator import add
from generateGraph import generateGraphNPP
import os
from networkx.algorithms import community
import pandas as pd
import copy
from random import random
import multiprocessing

def load_graph(filename, p_with, p_across,  group_ratio ,num_nodes,weight):
	try:
		f = open(filename+'.txt', 'r')
		print("loaded: " + filename )
		G = nx.Graph()
		n,m = map(int, f.readline().split())
		for i, line in enumerate(f):
			if i < n :
				node_str = line.split()
				u = int(node_str[0])
				group = int(node_str[1])
				G.add_node(u, group = group)
			else:
				edges_str = line.split()
				u = int(edges_str[0])
				v = int(edges_str[1])
				weight = float(edges_str[2])
				G.add_edge(u,v, weight = weight)
		f.close()       
    # Store configuration file values
	except FileNotFoundError:
		print(f"File not found at {filename}, building new graph...")
		G = generateGraphNPP(num_nodes, filename = filename +'.txt', p_with = p_with, p_across = p_across, group_ratio =group_ratio, weight= weight)

	return G

def graph_stats(G, num_groups, print_stats =True):

	# print average weights of each group
	w_across = []
	w_within = []
	num = []
	edges_w = []
	edges_a = []
	edges = 0
	for i in range(num_groups):
		w_within.append(0.0)
		w_across.append(0.0)
		num.append(0)
		edges_w.append(0)
		edges_a.append(0)


	for n, nbrs in G.adj.items():
		group = G.nodes[n]['group']
		
		num[group] += 1
		
		for nbr, eattr in nbrs.items():
			edges += 1
			group_2 = G.nodes[nbr]['group']

			if group == group_2: 
				edges_w[group] += 1
				w_within[group_2] += eattr['weight']
			else:
				w_across[group] += eattr['weight']
				edges_a[group] += 1

	#print(np.sum(np.asarray(edges_a)) / 2 + np.sum(np.asarray(edges_w)) / 2)
	# across edges could be a float since bothe the nodes are not in the same group so dividing it by 2 could creat a fraction i.e. each accross edge in a groups counts as 0.5 
	if print_stats:
		print( f"Total nodes: {np.sum(np.asarray(num))}, Edges {edges / 2 }" )
		print( f' Largest connected component is {len(max(nx.connected_components(G), key=len))}')
		for i in range(num_groups):
			print( f'Group: {i} nodes {num[i]}, edges_w: {edges_w[i] / 2}, edges_a: {edges_a[i] / 2}, w_wthn: {w_within[i] / edges_w[i]}, w_across: {w_across[i] / edges_a[i]}')

	
	return (num, edges_a, edges_w, w_within, w_across) 

def write_files(filename, num_influenced, seeds_grouped):
	'''
	write num_influenced is list of lists size is num seeds, and each element size is num groups
		  and seeds_grouped is list of list of lists  
	each row 
	I_1, I_2 ..  [seed_list_1 comma separated];[seed_list_2]; ...
	.
	.
	.
	'''
	f = open(filename +'_results.txt', 'w')
	for I,S in zip(num_influenced,seeds_grouped):
		f.write(f'{str(np.sum(np.asarray(I)))} ') # so the files are compatible with previous version as the previous version had total influence in the beginning 
		for i in I:
			f.write(f'{str(i)} ')

		for idx_out, grp_seeds in enumerate(S):
			for idx, seed_ids in enumerate(grp_seeds):
				if idx == len(grp_seeds) - 1:
					f.write(f'{seed_ids}')
				else:
					f.write(f'{seed_ids},')
			if idx_out != len(S) - 1:		
				f.write(';')
		f.write('\n')

	f.close()

def read_files(filename):
	'''
	returns num_influenced, num_influenced_a, num_influenced_b -> list 
		  and seeds_a list of lists i.e. actual id's of the seeds chosen 
		      seeds_b list of lists

	'''
	f = open(filename+'_results.txt', 'r')
	num_influenced =[]
	seeds_grouped =[]

	items = []
	for line in f:
		items = line.split()
		influenced = []
		for idx, i in enumerate(items[1:]):
			if idx < len(items) - 2: 
				influenced.append(float(i))

		seeds_group = items[-1].split(';')
		seed_interm = [] # holds groups 
		for seeds in seeds_group:
			
			seed_list = []
			if seeds != "":
				seed_list = list(map(int,seeds.split(',')))

			seed_interm.append(seed_list)

		num_influenced.append(influenced)
		seeds_grouped.append(seed_interm)
		
	f.close()
	return num_influenced, seeds_grouped

def write_paper_format(filename, num_influenced, seeds_grouped, population_group):

	#print( " start printing in " + filename)
	f = open(filename +'_results_group_influenced.txt', 'w')
	for I in num_influenced:
		f.write(f'{str(np.sum(np.asarray(I)))}\t') # so the files are compatible with previous version as the previous version had total influence in the beginning 
		for i in I:
			f.write(f'{str(i)}\t')
		f.write('\n')
	f.close()

	T = np.sum(np.asarray(population_group))
	#print( " start printing in " + filename)
	f = open(filename +'_results_group_fraction_influenced.txt', 'w')
	for index, I in enumerate(num_influenced):
		#print(f'{index+1}\t{str(np.sum(np.asarray(I))/T)}\t',end='')
		f.write(f'{index+1}\t{str(np.sum(np.asarray(I))/T)}\t') # so the files are compatible with previous version as the previous version had total influence in the beginning 
		for idx, i in enumerate(I):
			f.write(f'{str(i/population_group[idx])}\t')
			#print(f'{str(i/population_group[idx])}\t',end='')
		f.write('\n')
		#print('\n',end='')
	f.close()	

	f = open(filename +'_results_group_population.txt', 'w')
	#print( " population ", population_group)
	f.write(f'{str(np.sum(np.asarray(population_group)))} ')
	for p in population_group:
		f.write(f'{str(p)}\t') # so the files are compatible with previous version as the previous version had total influence in the beginning 
		#f.write('\n')
	f.close()

	f = open(filename +'_results_group_seeds.txt', 'w')
	for S in seeds_grouped:
		for idx_out, grp_seeds in enumerate(S):
			f.write(f'{len(grp_seeds)}\t')	
	f.close()

def plot_influence(influenced_group, num_seeds_group, filename, population_group):

	total_influenced = []
	num_seeds = 0 
	num_groups = 0 
	total_pop = np.sum(np.asarray(population_group))
	seeds_per_group = []
	for s in num_seeds_group[-1]: # num_seeds_group [ [ [ ] - > seed id for group i ( this is what we sum ) ...  ]-> len num groups ....]-> len is num_seeds
		num_groups += 1
		num_seeds += len(s) # s i also a list as num_seeds_group is list of list of lists 

	for i in range(num_groups):
		seeds_per_group.append([])

	for s in num_seeds_group:
		#seed_dist = []
		for idx, k in enumerate(s): # s is len is group number
			seeds_per_group[idx].append(len(k) + 0.05) # k is the ids of the seeds from each group

		#seeds_per_group.append(seed_dist)

	for infl in influenced_group: # influenced group is list of lists [ [ Ig1 ] -> len is num groups , [ ] ]-> len = seeds 
		total_influenced.append(np.sum(np.asarray(infl)))

	print( 'num seeds ', num_seeds)
	# total influence
	fig = plt.figure(figsize=(6,4))
	plt.plot(np.arange(1, num_seeds + 1), total_influenced,'g+')
	plt.xlabel('Number of Seeds')
	plt.ylabel('Total Influenced Nodes')
	#plt.legend(loc='best')
	plt.savefig(filename + '_total_influenced.png',bbox_inches='tight')
	plt.close(fig)

	# total influence fraction 
	fig = plt.figure(figsize=(6,4))
	plt.plot(np.arange(1, num_seeds + 1),np.asarray(total_influenced)/(total_pop),'g+')
	plt.xlabel('Number of Seeds')
	plt.ylabel('Total Fraction Influenced Nodes')
	#plt.legend(loc='best')
	plt.savefig(filename + '_total_fraction_influenced.png',bbox_inches='tight')
	plt.close(fig)
	# group wise influenced
	fig = plt.figure(figsize=(6,4))
	for idx, p in enumerate(population_group):
		plt.plot(np.arange(1, num_seeds + 1), np.asarray(influenced_group)[:,idx], label= f'Group {idx}')

	
	plt.xlabel('Number of Seeds')
	plt.ylabel('Total Influenced Nodes')
	plt.legend(loc='best')
	plt.savefig(filename + '_group_influenced.png',bbox_inches='tight')
	plt.close(fig)

	# fraction group influenced 
	fig = plt.figure(figsize=(6,4))
	for idx, p in enumerate(population_group):
		plt.plot(np.arange(1, num_seeds + 1), np.asarray(influenced_group)[:,idx] / p, label= f'Group {idx}')
	plt.xlabel('Number of Seeds')
	plt.ylabel('Fraction of Influenced Nodes')
	plt.legend(loc='best')
	plt.savefig(filename + '_fraction_group_influenced.png',bbox_inches='tight')
	plt.close(fig)

	# Seeds group memeber ship 
	#fig = plt.figure(figsize=(6,4))
	fig, ax = plt.subplots(figsize=(8,4))
	bar_width = 0.35
	index = np.arange(1, num_seeds + 1)
	rects = []
	for idx in range(num_groups): 
		print(seeds_per_group[idx]) # this has group representation for given seed set i.e among 1st seed how many are from group idx 

		rects.append(plt.bar(3 * index  + bar_width * (idx), seeds_per_group[idx], bar_width,
	                label = f'Group {idx}'))

	plt.legend(loc='best')
	plt.xlabel('Total Number of Seeds')
	plt.ylabel('Number of Seeds from each group')
	plt.title('Seed distribution in groups')
	plt.xticks(3 * index + (bar_width * num_groups)/2, index)

	# plt.plot(np.arange(1, num_seeds + 1), num_seeds_a / num_seeds, 'r+')
	# plt.plot(np.arange(1, num_seeds + 1), num_seeds_b / num_seeds, 'b.')
	# plt.xlabel('Total Number of Seeds')
	# plt.ylabel('Number from each group')
	plt.savefig(filename + '_seed_groups.png', bbox_inches='tight')
	plt.close(fig)	#

def plot_influence_diff(influenced_a_list, influenced_b_list, num_seeds, labels, filename, population_a, population_b):
	'''
	list of lists influenced_a and influenced_b
	'''
	fig, ax = plt.subplots(figsize=(8, 6), dpi= 80)
	index = np.arange(1, num_seeds + 1)
	for i, (influenced_a, influenced_b) in enumerate(zip (influenced_a_list, influenced_b_list)):
		ax.plot(index, (np.asarray(influenced_a) + np.asarray(influenced_b))/(population_a + population_b), label=labels[i], ls= '-', alpha=0.5)
		

	legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
	plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


	
	plt.xlabel('Number of Seeds')
	plt.ylabel('Fraction of Influenced Nodes (F(S))')
	plt.savefig(filename+'_total_influenced.png',bbox_inches='tight')
	plt.close(fig)

	# comparison abs difference 
	fig, ax = plt.subplots(figsize=(8, 6), dpi= 80)
	index = np.arange(1, num_seeds + 1)
	for i, (influenced_a, influenced_b) in enumerate(zip(influenced_a_list, influenced_b_list)):
		ax.plot(index, np.abs(np.asarray(influenced_a)/population_a - np.asarray(influenced_b)/population_b), label=labels[i], ls= '-', alpha=0.5)
		

	legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
	plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


	
	plt.xlabel('Number of Seeds')
	plt.ylabel('Absolute difference of Influenced Nodes (|Fa - Fb|)')
	plt.savefig(filename+'_difference_total_influenced.png',bbox_inches='tight')
	plt.close(fig)

def load_random_graph(filename, n,p,w):
	#return get_random_graph(filename+'.txt', n,p,w)
	'''
	Random graph, used it for testing discounted timings submodularity
	'''
	try:
		f = open(filename+'.txt', 'r')
		G = nx.Graph()
		n,m = map(int, f.readline().split())
		print("loaded: " + filename )
		for i, line in enumerate(f):
			if i < n :
				node_str = line.split()
				u = int(node_str[0])
				color = node_str[1]
				G.add_node(u, color = color)
			else:
				edges_str = line.split()
				u = int(edges_str[0])
				v = int(edges_str[1])
				weight = float(edges_str[2])
				G.add_edge(u,v, weight = weight)
		f.close()   
	except FileNotFoundError:
		print(f"File not found at {filename}, building new graph...")
		G = get_random_graph(filename+'.txt', n,p,w)
	return G 

def save_graph(filename,G):
	if filename:
		with open(filename, 'w+') as f:
			f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
			for n, ndata in G.nodes(data=True):
				f.write('%s %s%s'%(n, ndata['color'], os.linesep))
			for v1,v2,edata in G.edges(data=True):
                #for it in range(edata['weight']):
				f.write('%s %s %s%s'%(v1, v2, edata['weight'], os.linesep))
		print("saved")

def get_random_graph(filename,n,p,w):
	
	G = nx.binomial_graph(n, p)
	color = 'blue' # all nodes are one color 
	nx.set_node_attributes(G, color, 'color')
	nx.set_edge_attributes(G, w, 'weight')
	
	#save_graph(filename, G)

	return G 

def get_twitter_data(filename,w = None, save = False	):
	'''
	reads twitter data, makes bipartition and assign group memebership 
	with constant weights of infection 
	'''
	f = None
	DG = None
	try:
		f = open(filename+'.txt', 'r')
		print("loaded: " + filename )
		DG = nx.DiGraph()
		n,m = map(int, f.readline().split())
		for i, line in enumerate(f):
			if i < n :
				node_str = line.split()
				u = int(node_str[0])
				color = node_str[1]
				DG.add_node(u, color = color)
			else:
				edges_str = line.split()
				u = int(edges_str[0])
				v = int(edges_str[1])
				weight = float(edges_str[2])
				if w is not None:
					DG.add_edge(u,v, weight = w)
				else: 
					DG.add_edge(u,v, weight = weight)
		f.close() 
		
	except FileNotFoundError:
		#
		print(" Making graph ") 
		f = open('twitter/twitter_combined.txt', 'r')
		DG = nx.DiGraph()

		for line in f:
		    node_a, node_b = line.split()
		    DG.add_nodes_from([node_a,node_b])
		    DG.add_edges_from([(node_a, node_b)])

		    DG[node_a][node_b]['weight'] = w 
		
		print("done with edges and weights ")

		G_a , G_b = community.kernighan_lin_bisection(DG.to_undirected())
		for n in G_a:
			DG.nodes[n]['color'] = 'red'
		for n in G_b:
			DG.nodes[n]['color'] = 'blue'

		save_graph(filename, DG)
	 

	return DG

def order_seeds(G,S,num_groups):
	
	ordered_seeds = []
	for i in range(num_groups):
		ordered_seeds.append([])

	for n in S:
		ordered_seeds[G.nodes[n]['group']].append(n)

	return ordered_seeds

def get_results_budget(filename, budget, tau, gamma, beta, type_algo, G, num_groups):

	results_exist = True
	S = [] # set of selected nodes
    #S_hard = [2078,107,1912,1431,102,2384,1596,1322,1006,821,1211,1684,1833,3852,2351]
	influenced_grouped = [] # is a list of lists [[ I_g1(S1),I_g2(S1)....I_gn(S1)][ I_g_1(S1,S2) ..... ]
	seeds_grouped = [] # is list of lists of lists  [[[S_g1(among S1)],[S_g2( among S1)]....] S_g_1(among S1,S2) ..... ]
   
	seed_range = []
	if  type_algo == 1:
		filename = filename + f'_lazy_greedy_tau_{tau}_'

	elif type_algo == 2:
		filename = filename + f'_lazy_greedy_log_gamma_{gamma}_tau_{tau}_'

	elif type_algo == 3:
		filename = filename + f'_lazy_greedy_root_gamma_{gamma}_beta_{beta}_tau_{tau}_'

	

	stats = graph_stats(G, num_groups, print_stats = False)
	print(" in node thingy ")
	try :

		influenced_grouped, seeds_grouped = read_files(filename)# change this to multiple groups
        # for I in influenced_grouped: 
        #     print(I)
            
		write_paper_format(filename, influenced_grouped, seeds_grouped, stats[0])
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
            #print()
			plot_influence(influenced_grouped, seeds_grouped, filename, stats[0])

			return (results_exist, filename, seed_range, S, influenced_grouped, seeds_grouped)
		else:
			seed_range = range(budget - len(S))

	except FileNotFoundError:
		print( f'{filename} not Found ')
		seed_range = range(budget)
		results_exist = False

	return results_exist, filename, seed_range, S, influenced_grouped, seeds_grouped

def get_results_cover(filename, G, budget, tau, type_algo, num_groups):


	stats = graph_stats(G, num_groups, print_stats = False)
	results_exist = True

	influenced_grouped = []
	seeds_grouped = []
	if  type_algo == 1:
		filename = filename + f'lazy_set_cover_reach_{budget}_tau_{tau}_no_groups_'
		budget = (budget/num_groups) # as we multiply the with num_groups 
	elif type_algo == 2:
		filename = filename + f'lazy_set_cover_reach_{budget}_tau_{tau}_'
	

	try:
		influenced_grouped,seeds_grouped = read_files(filename)

		print( f' Found file {filename} ******* \n \n ')
		write_paper_format(filename, influenced_grouped, seeds_grouped,stats[0])

		return (results_exist, filename, budget, influenced_grouped, seeds_grouped)
        
	except FileNotFoundError:
		print( f'{filename} not Found ')
		results_exist = False

	return (results_exist, filename, budget, influenced_grouped, seeds_grouped)

def get_batch(s, batch_size):
        
    nodes_to_return = []
    nodes = []
    i = 0
    s_ = copy.deepcopy(s)
    #print(f'length pq {len(s_.pq)}')
    count = 0
    while True:
        
        try:
            n, priority = s_.pop_item()
        except Exception as e:
            print(f'popped the pq empty')
            if nodes:
                nodes_to_return.append(nodes)
            break
        nodes.append(n)
        i+=1
        if i == batch_size:
            nodes_to_return.append(nodes)
            nodes = []
            i = 0
        count+=1
              
    print(f'popped {count} items')
    return nodes_to_return

def get_seed_sample(nodes, sample_size, G, filename, print_stats = True):
    
	nodes_to_return = []
	from config import infMaxConfig
	c = infMaxConfig()
	#print(c.filename)
	try:
		f = open(c.filename+f'_reduced_node_{sample_size}.txt', 'r')
		for line in f:
			item = int(line.strip())
			nodes_to_return.append(item)
		print( f' file: {c.filename}_reduced_node_{sample_size}.txt found')
		f.close()
	except FileNotFoundError:

		print( f'{c.filename}_reduced_node_{sample_size}.txt not Found ')
		
		nodes_array = np.asarray(nodes)
		nodes_to_return = np.random.choice(nodes_array, sample_size, replace = False)
		nodes_to_return = nodes_to_return.tolist()

		f = open(c.filename +f'_reduced_node_{sample_size}.txt', 'w')
		for n in nodes_to_return:
			f.write(f'{str(n)}\n')
		f.close()

	men = 0
	for n in nodes_to_return:
		if G.nodes[n]['group'] == 0:
			men+=1

	if print_stats:
		print(f'Subsample seed set Men: {men} Women: {sample_size - men}')

	return nodes_to_return

def get_size(s, node = -1):
    
        s_ = copy.deepcopy(s)
        nodes = copy.deepcopy(s_.pq)

        print(f' in get size length of pq {len(s_.pq)}')
        count = 0
        while True:
            
            try:
                n, priority = s_.pop_item()
                #nodes.remove(n)
                if n == node:
                    print(f' found the node recently added {node} ....')

                for node_item in nodes:
                    if node_item[-1] == n:
                        nodes.remove(node_item)
            except Exception as e:
                print(f'popped the pq empty')
                break
            count+=1
                  
        print(f'in get size popped {count} items')
        print(f'remaining nodes are {nodes}')
        print('....................................')

def write_averages_variance(averages, variances, filename):
	f = open(filename + "averages.txt", 'w')

	for a in averages:
		for a_ in a:
			f.write(f'{str(a_)}\t')
		f.write('\n')
	f.close()

	f = open(filename + "variances.txt", 'w')
	for v in variances:
		for v_ in v:
			f.write(f'{str(v_)}\t')
		f.write('\n')

	f.close()

def gen_graphs(G):

	#G,runs = inp 

	#for i in range(runs):
	G_ = nx.Graph()
	for e in G.edges():
		w = G[e[0]][e[1]]['weight'] # probability of infection 

		if random() <= w:
			G_.add_nodes_from([e[0],e[1]]) # adding node multiple times doesn't double count 
			G_.nodes[e[0]]['group'] = G.nodes[e[0]]['group']
			G_.nodes[e[1]]['group'] = G.nodes[e[1]]['group']
			G_.add_edges_from([(e[0],e[1])])

	del(G)
	return G_

def generate_graphs(G,runs):

	graphs = []

	pool = multiprocessing.Pool(int(multiprocessing.cpu_count() -1 ))

	graphs = pool.map(gen_graphs, (G for i in range(runs)))

	pool.close()
	pool.join()
	
	

	return graphs





















































