import networkx as nx

def obtain_topology(**kwargs):
  # # Call a specific topology as follows:
  # topology = obtain_topology(name='wheel', nodes=12)
  # topology = obtain_topology(name='balanced_tree', nodes=12, h=1)
  # topology = obtain_topology(name='complete_graph', nodes=12)
  # topology = obtain_topology(name='circular_ladder_graph', nodes=int(12 / 2))  # only for even number of nodes
  # topology = obtain_topology(name='cycle_graph', nodes=12)
  # topology = obtain_topology(name='ladder_graph', nodes=int(20 / 2))  # only for even number of nodes
  # topology = obtain_topology(name='grid_graph', dim=(3, 3))  # only for even number of nodes
  # topology = obtain_topology(name='grid_graph', dim=(3, 2, 3))  # only for even number of nodes
  # topology = obtain_topology(name='hypercube_graph', nodes=4)  # only for even number of nodes
  # topology = obtain_topology(name='watts_strogatz_graph', nodes=10, k=2, p=0.5)  # only for even number of nodes

  for key in kwargs:
    if kwargs[key] == 'wheel':
      graph_object = nx.wheel_graph(kwargs['nodes']).to_directed() # To directed is to get a digraph
    if kwargs[key] =='balanced_tree':
      graph_object = nx.balanced_tree(kwargs['nodes'], kwargs['h']).to_directed() # To directed is to get a digraph
    if kwargs[key] =='complete_graph':
      graph_object = nx.complete_graph(kwargs['nodes']).to_directed() # To directed is to get a digraph
    if kwargs[key] =='circular_ladder_graph':
      graph_object = nx.circular_ladder_graph(kwargs['nodes']).to_directed() # To directed is to get a digraph
    if kwargs[key] =='cycle_graph':
      graph_object = nx.cycle_graph(kwargs['nodes']).to_directed() # To directed is to get a digraph
    if kwargs[key] =='ladder_graph':
      graph_object = nx.ladder_graph(kwargs['nodes']).to_directed() # To directed is to get a digraph
    if kwargs[key] =='grid_graph':
      graph_object = nx.grid_graph(kwargs['dim']).to_directed() # To directed is to get a digraph
    if kwargs[key] =='hypercube_graph':
      graph_object = nx.hypercube_graph(kwargs['nodes']).to_directed() # To directed is to get a digraph
    if kwargs[key] =='watts_strogatz_graph': # Small world graph # https://networkx.org/documentation/networkx-1.9/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html
      graph_object = nx.watts_strogatz_graph(kwargs['nodes'], kwargs['k'], kwargs['p']).to_directed()
    nx.set_edge_attributes(graph_object, values=1, name='weight')  # all weights to 1
  return graph_object
