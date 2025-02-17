import networkx as nx
import random
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def generateGraph (n,m,filename='',pw=.75,maxw=5):
    G = nx.dense_gnm_random_graph(n,m)
    for e in G.edges():
        if random.random() < pw:
            G[e[0]][e[1]]['weight'] = 1
        else:
            G[e[0]][e[1]]['weight'] = random.randint(2,maxw)
    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
            for v1,v2,edata in G.edges(data=True):
                for it in range(edata['weight']):
                    f.write('%s %s%s' %(v1, v2, os.linesep))
    return G

def generateGraph_ours (n,m,filename='',p_cliq =.75):
    
    #DG = nx.DiGraph()
    G = nx.Graph()
    nodes_a = []
    nodes_b = []
    for i in np.arange(n):

        toss = np.random.random_sample()
        if toss >= 0.7:
            G.add_nodes_from(i, color = 'red', active = 0, t = 0)
            nodes_a.append(i)
        else:
            G.add_nodes_from(i, color = 'blue', active = 0, t = 0)
            nodes_b.append(i)

    
    i = 0 
    while i < m:
        Y = np.random.binomial(1, p_cliq, 1)
        n_1 = np.random.randint(0,n)
        
        if n_1 in nodes_a:
            if Y == 1:
                n_2 = nodes_a[np.random.randint(0, len(nodes_a))]

            else:
                n_2 = nodes_b[np.random.randint(0, len(nodes_b))]
        else:
            if Y == 1:
                n_2 = nodes_b[np.random.randint(0, len(nodes_b))]
            else:
                n_2 = nodes_a[np.random.randint(0, len(nodes_a))]

        if G.has_edge(n_1,n_2) or n_1 == n_2:
            continue 

        G.add_edges_from([(n_1,n_2)])
        a = 0.0 
        b = 0.5
        G[n_1][n_2]['weight'] = (b-a) * np.random.random_sample() + a
        i+=1

    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
            for v1,v2,edata in G.edges(data=True):
                for it in range(edata['weight']):
                    f.write('%s %s%s' %(v1, v2, os.linesep))
    return G 

def generateGraphNPP(num_nodes,filename='',p_with =.75, p_across = 0.1, group_ratio = 0.7, weight = 0.05):
    
    #DG = nx.DiGraph()
    G = nx.Graph()
    
    for i in np.arange(num_nodes):

        G.add_node(i, t = 0)

        # this might give slightly different fractions sometimes 
       #  toss = np.random.uniform(0,1.0,1)[0]
       # #print(i)
       #  if toss <= group_ratio:
       #      G.add_node(i, group = 0, active = 0, t = 0)
            
       #  else:
       #      G.add_node(i, group = 1, active = 0, t = 0)
    G0 = np.random.choice(num_nodes-1, int(group_ratio * num_nodes), replace=False)

    for n in G.nodes():
        if n in G0:
            G.node[n]['group'] = 0 
        else: 
            G.node[n]['group'] = 1
    num_edges = 0 
    for i in np.arange(num_nodes):
        for j in np.arange(num_nodes):
            if G.has_edge(i,j) or i == j:
                continue 

            if G.nodes[i]['group'] == G.nodes[j]['group']:
                Y = np.random.binomial(1, p_with, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i,j)])
                    G[i][j]['weight'] = weight# np.random.uniform(0,0.1,1)[0]
                    num_edges +=1

            else:
                Y = np.random.binomial(1, p_across, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i,j)])
                    G[i][j]['weight'] = weight#np.random.uniform(0,.1,1)[0]
                    num_edges +=1 

    print(f'number of edges: {num_edges}')
        

    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
            for n, ndata in G.nodes(data=True):
                f.write('%s %s%s'%(n, ndata['group'], os.linesep))
            for v1,v2,edata in G.edges(data=True):
                #for it in range(edata['weight']):
                f.write('%s %s %s%s'%(v1, v2, edata['weight'], os.linesep))

    return G 

# outer chain
def chain(l,curr_num,p):
    groups = {}
    nodes = np.arange(curr_num,curr_num + l)
    edges = []
    for n in nodes:#add grou1 nodes
        groups[n] = 0
        
   
    nodes_2 = np.arange(curr_num+l,curr_num + 2*l)
    for n in nodes_2:# add group 2 nodes 
        groups[n] = 1
        nodes = np.append(nodes, [n])
    #edges.append((curr_num - 1 ,nodes[0], {'w': p}))
    
    for i in range(len(nodes) - 1) : # add edges 
        edges.append((nodes[i] , nodes[i+1], {'weight': p}))
    #print(f'Nodes are {nodes}, group: {groups}, edges: {edges}')
    return nodes, edges, groups

def generate_example(l,p):
    G = nx.Graph()
    v1 = 0
    G.add_node(v1, group = 0)
    curr_len = len(G.nodes())

    for i in range(curr_len, curr_len + 4 * l): # middle chain 
        G.add_node(i, group = 0)
        G.add_edge(i-1,i, weight = p)

    v2 = len(G.nodes())

    G.add_node(v2, group = 0)
    G.add_edge(v2-1 , v2, weight = p)
    curr_len = len(G.nodes())
    #print(f'Nodes are {G.nodes()}')
    nodes , edges, groups = chain(l, curr_len,p)

    #print(G.nodes())
    # print(edges)
    # 
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    for n in nodes:
        G.nodes[n]['group'] = groups[n]
    #print(nodes, edges)
    G.add_edge(v2,nodes[0],weight = p)

    curr_len = len(G.nodes())
    nodes , edges, groups = chain(l, curr_len,p)

    #print(G.nodes())
    # print(edges)
    # 
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    for n in nodes:
        G.nodes[n]['group'] = groups[n]
    #print(nodes, edges)
    G.add_edge(v2,nodes[0],weight = p)


    curr_len = len(G.nodes())
    nodes , edges, groups = chain(l, curr_len,p)

    #print(G.nodes())
    # print(edges)
    # 
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    for n in nodes:
        G.nodes[n]['group'] = groups[n]
    #print(nodes, edges)
    G.add_edge(v1,nodes[0], weight = p)



    curr_len = len(G.nodes())
    nodes , edges, groups = chain(l, curr_len,p)

    #print(G.nodes())
    # print(edges)
    # 
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    for n in nodes:
        G.nodes[n]['group'] = groups[n]
    #print(nodes, edges)
    G.add_edge(v1,nodes[0], weight = p)

    nx.draw(G,with_labels = True)
    plt.savefig("Example_Graph.png")

    return G, v1, v2 

if __name__ == '__main__':
    generateGraph(30, 120, 'small_graph.txt')