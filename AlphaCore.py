import sqlite3 as sql
import networkx as nx
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# alphaCore ranking of nodes in a complex, directed network
#
# \code{alphaCore} returns a node ranking of a graph
#
# Iteratively computes a node ranking based on a feature set derived from
# edge attributes and optionally static node features using the
# mahalanobis data depth function at the origin.
#
# @param graph A networkx directed graph
# @param stepSize defines the stepsize of each iteration as percentage of node count
# @param startEpsi the epsilon to start with. Removes all nodes with depth>epsilon at start
# @param expoDecay dynamically reduces the step size, to have high cores with few nodes if true
# @return A dataframe of node name and alpha value indicating the ranking
def alphaCore(graph, stepSize, startEpsi, expoDecay):
    #1
    data = computeNodeFeatures(graph)
    #2 #3 calculate the Mahalanobis depth and add it to the respective row of the dataframe
    data['mahal'] = calculateMahal(y=data[['inDegree', 'inStrength']], data=data[['inDegree', 'inStrength']])
    #4
    epsi = startEpsi
    #5
    core = {}
    #6
    batch = []
    #7
    alpha = 1 - epsi
    #8
    alphaPrev = alpha
    #9
    batchID = 0
    #10
    while graph.number_of_nodes() > 0:
        #11
        while True:
            depthFound = False  # to simulate do-while loop; used to check if there exists a node with depth >= epsi on current iteration
            #12
            for index, row in data.iterrows():
                if row['mahal'] >= epsi:
                    depthFound = True
                    #13
                    core[row['nodeID']] = alphaPrev  # set node core
                    #14
                    batch.append(batchID)
                    #15
                    graph.remove_node(row['nodeID'])
            #16
            batchID += 1
            #19 while condition of do-while loop of #11
            if graph.number_of_nodes() == 0 or not depthFound:
                break
            #17
            data = computeNodeFeatures(graph)  # recompute node properties
            #18
            data['mahal'] = calculateMahal(y=data[['inDegree', 'inStrength']], data=data[['inDegree', 'inStrength']])  # recompute depth
        #20
        alphaPrev = alpha
        #21
        if expoDecay and graph.number_of_nodes() > 0:  # exponential decay
            localStepSize = math.ceil(graph.number_of_nodes() * stepSize)
            data.sort_values(ascending=False, by=['mahal'])
            epsi = data.at[localStepSize - 1, 'mahal']
        else:  # step decay
            epsi -= stepSize
        #22
        alpha = 1 - epsi
    #23
    return [core, batch]


# computes the node features of a given directed graph and returns a dataframe containing the features of each node
def computeNodeFeatures(graph):
    nodeID = []
    inDegree = []
    outDegree = []
    inStrength = []
    outStrength = []
    for node in graph:
        nodeID.append(node)
        inEdges = graph.in_edges(node)
        outEdges = graph.out_edges(node)
        edgeAttributes = nx.get_edge_attributes(graph, "value")
        inStrengthNode = 0
        outStrengthNode = 0
        for edge in inEdges:
            inStrengthNode += edgeAttributes[edge]
        for edge in outEdges:
            outStrengthNode += edgeAttributes[edge]
        inDegree.append(graph.in_degree(node))
        outDegree.append(graph.out_degree(node))
        inStrength.append(inStrengthNode)
        outStrength.append(outStrengthNode)

    # currently only adding inDegree and inStrength to dataframe
    nodeFeat = {"nodeID": nodeID, "inDegree": inDegree, "inStrength": inStrength}
    df = pd.DataFrame(nodeFeat, columns=['nodeID', 'inDegree', 'inStrength'])
    return df


# computes the mahalanobis depth of a given set of data and returns the diagonal
def calculateMahal(y, data):
    y_mu = y - np.mean(data)
    cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()


#########################################  Aux Functions #########################################


def graphToFigure(graph):
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color = 'r', node_size = 100, alpha = 1)
    ax = plt.gca()
    for e in graph.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                    ),
                                    ),
                    )
    plt.axis('off')

    print("Saving figure")
    plt.savefig("StableCoinGraph.png")




#########################################  Test Alphacore Script #########################################

conn1 = sql.connect('AlphaTest.db')
c1 = conn1.cursor()
c1.execute("SELECT * FROM token_transfers")
items = c1.fetchall()
G = nx.DiGraph()

count = 0
for item in items:
    G.add_edge(item[0], item[1], value=item[2])
    count += 1
    if count == 1000: # terminate building of graph at 1000 vertices
        break

print("Graph Made")

alph = alphaCore(G, 0.1, 0.1, False)
print("Alphacore function completed")

print(alph)
