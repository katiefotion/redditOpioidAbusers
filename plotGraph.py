'''
Created on 06-May-2017

@author: Neharika Mazumdar
'''

import networkx as nx
import matplotlib.pyplot as plt

H=nx.Graph()

#drawing the graph with any standard layout 
def draw():
    pos = nx.spring_layout(H)
    nx.draw_networkx_nodes(H, pos=pos, nodelist = H.nodes())
    nx.draw_networkx_edges(H, pos=pos, edgelist = H.edges())
    nx.draw_networkx_labels(H, pos=pos)
    nx.draw_random(H)
    plt.show()

#adding nodes and edges to the graph    
def addConnections(activeUsersList,usersDict,subR):
    H.add_nodes_from(activeUsersList);
    for key,value in usersDict.iteritems():
        H.add_node(value)
        H.add_edge(key, value)
        
   
   

    