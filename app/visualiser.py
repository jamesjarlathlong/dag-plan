import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import brewer2mpl
cm = brewer2mpl.get_map('RdYlBu', 'Diverging', 9).mpl_colormap
def label_weight(d):
    return {k: {'weight':v} for k,v in d.items()}
def rssi_to_nx(rssi):
    labeled = {k:label_weight(v) for k,v in rssi.items()}
    return nx.DiGraph(labeled)
# Add a dark background
def plotgraph(graph):
    sns.set_style('white')
    # Initialize figure
    fig = plt.figure(figsize = (8, 8))
    ax  = fig.add_subplot(1, 1, 1)
    #G = nx.convert.convert_to_directed(nx.barabasi_albert_graph(30,5))
    G = rssi_to_nx(graph)
    # Create force-directed layout for the graph
    pos = nx.spring_layout(G)
    xs = np.array([v[0] for k, v in pos.items()])
    ys = np.array([v[1] for k,v in pos.items()])
    pos_array = np.vstack((xs,ys)) # Format into a NumPy array
    # Define node colors based on in-degree
    node_color = np.array([len(G.predecessors(v)) for v in G], dtype=float)
    pr = nx.pagerank(G)
    sizes = np.array([1e4*v for k,v in pr.items()])#[v for k,v in .items()]
    # Plot each edge
    for i in G:
        for j in G[i]:
            ax.plot(pos_array[0, [i, j]], pos_array[1, [i, j]], color='#525252',linewidth=1)
    # Plot nodes
    ax.scatter(pos_array[0], pos_array[1], sizes, color = cm(node_color/node_color.max()), zorder = 10)
    for i in G:
        ax.annotate('   '+str(i), xy=(pos_array[0,i], pos_array[1,i]), color ='#252525',zorder=11)
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    sns.despine(left=True,right=True, top=True,bottom=True)
    return fig, ax