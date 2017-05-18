
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
def addConnections(usersDict):
    H.add_nodes_from(list(usersDict.keys()));
    for key,value in usersDict.iteritems():
        commentUsersList=value
        for item in commentUsersList:
            H.add_node(item)
            H.add_edge(key, item)
        
   
   

    