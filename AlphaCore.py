import sqlite3 as sql
import networkx as nx
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# alphaCore ranking of nodes in a complex, directed network
#
# Iteratively computes a node ranking based on a feature set derived from
# edge attributes and optionally static node features using the
# mahalanobis data depth function at the origin.
#
# @param graph A networkx directed graph
# @param stepSize Defines the stepsize of each iteration as percentage of node count
# @param startEpsi The epsilon to start with. Removes all nodes with depth>epsilon at start
# @param expoDecay Dynamically reduces the step size, to have high cores with few nodes if true
# @return A dataframe of columns nodeID, alpha value, and batchID
def alphaCore(graph, stepSize, startEpsi, expoDecay):
    #1
    data = computeNodeFeatures(graph)
    #2 #3 calculate the Mahalanobis depth and add it to the respective row of the dataframe
    data['mahal'] = calculateMahalFromCenter(data, 0)
    #4
    epsi = startEpsi
    #5
    node = []
    alphaVals = []
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
                    print(row)
                    depthFound = True
                    #13
                    node.append(row['nodeID'])  # set node core
                    alphaVals.append(alphaPrev)
                    #14
                    batch.append(batchID)
                    #15
                    graph.remove_node(row['nodeID'])
            #16
            batchID += 1
            #19 while condition of do-while loop of #11
            if graph.number_of_nodes() == 0 or not depthFound:
                break
            # if graph.number_of_nodes() > 1:  # prevent error when trying to compute depth when only one node remains
            #17
            data = computeNodeFeatures(graph)  # recompute node properties
            #18
            data['mahal'] = calculateMahalFromCenter(data, 0)  # recompute depth
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
    return pd.DataFrame({'nodeID': node, 'alpha': alphaVals, 'batchID': batch})


# Computes the node features of a given directed graph and returns a dataframe containing the features of each node
#
# @param graph A networkx directed graph
# @return A dataframe containing the computed node features with each row as a new entry and columns as different features
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
    # df = pd.DataFrame({"nodeID": nodeID, "inDegree": inDegree, "inStrength": inStrength})

    # currently adding inDegree, outDegree, inStrength, and outStrength to dataframe
    df = pd.DataFrame({"nodeID": nodeID, "inDegree": inDegree, "inStrength": inStrength, "outDegree": outDegree, "outStrength": outStrength})
    return df



# Computes the mahalanobis depth of each row of a given set of data and returns it as an array
#
# @param data Dataframe where each row is a new entry and each column after the first (nodeID) is a type of data
# @param center A center value calculated with respect to when computing mahalanobis depth
# @return An array containing the mahalanobis depth of each row entry of a given set of data
def calculateMahalFromCenter(data, center):
    matrix = data.drop("nodeID", axis=1)  # convert dataframe to numeric matrix by removing first column containing nodeID
    # print(matrix)
    x_minus_center = matrix.values - center
    # print(x_minus_center)
    x_minus_center_transposed = (matrix.values - center).T
    # print(x_minus_center_transposed)
    cov = np.cov(matrix.values.T)
    # print(cov)
    inv_cov = np.linalg.inv(cov)
    # print(inv_cov)
    left = np.dot(x_minus_center, inv_cov)
    mahal = np.dot(left, x_minus_center_transposed)
    return np.diagonal(np.reciprocal(1+mahal))  # diagonal contains the depth vaulues corresponding to each row from matrix



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
    # if item[1] not in G.nodes:
    #     G.add_node(item[1])
    # if item[2] not in G.nodes:
    #     G.add_node(item[2])
    G.add_edge(item[0], item[1], value=item[2])
    count += 1
    if count == 30: # terminate building of graph at count vertices
        break

print("Graph Made")

alph = alphaCore(G, 0.1, 1, False)
print("Alphacore function completed")

print(alph)

### Test calculateMahalFromCenter function ###
# nodeFeat = computeNodeFeatures(G)
# print(nodeFeat)
# nodeFeat['mahal'] = calculateMahalFromCenter(nodeFeat, 0)
# print('mahal')
# print(nodeFeat)
