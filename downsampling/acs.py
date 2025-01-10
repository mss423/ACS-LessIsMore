import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from vertex_embed import get_embeddings_task

# labels maps i to its label.
def build_graph(cos_sim, sim_thresh=0.0, max_degree=None, labels=None):
    G = nx.Graph()
    for i in range(len(cos_sim)):
        G.add_node(i)
        # Sort neighbors by similarity in descending order
        neighbors = sorted(enumerate(cos_sim[i]), key=lambda x: x[1], reverse=True)
        for j, similarity in neighbors:
            if j == i:
                continue
            if max_degree and G.degree(i) >= max_degree:
                break  # Exit the inner loop if max_degree is reached
            if similarity >= sim_thresh and labels and labels[i]==labels[j]:
                G.add_edge(i, j, weight=similarity)
        # add self-loop, doesn't count toward max_degree
        G.add_edge(i, i, weight=1)
    return G

# Graph sampling algorithms (max-cover)
def max_cover_sampling(graph, k):
    nodes = list(graph.nodes())
    selected_nodes = set()
    covered_nodes = set()

    for _ in range(k):
      if not nodes:
        break
      max_cover_node = max([node for node in nodes if node not in covered_nodes], key=lambda n: len(set(graph.neighbors(n)) - covered_nodes))
      selected_nodes.add(max_cover_node)
      # covered_nodes.add(max_cover_node)
      covered_nodes.update(graph.neighbors(max_cover_node))

      # Remove neighbors of selected node
      for neighbor in graph.neighbors(max_cover_node):
        if neighbor in nodes:
          nodes.remove(neighbor)
    return list(selected_nodes), len(nodes)

def calculate_similarity_threshold(data, num_samples, coverage, cap=None, epsilon=None, labels=None, sims=[707,1000]):
    total_num = len(data)
    if epsilon is None:
        # There is a chance that we never get close enough to "coverage" to terminate
        # the loop. I think at the very least we should have epsilon > 1/total_num.
        # So let's set set epsilon equal to the twice of the minimum possible change
        # in coverage.
        epsilon = 5 * 10 / total_num  # Dynamic epsilon

    if coverage < num_samples / total_num:
        node_graph = build_graph(data, 1)
        samples, rem_nodes = max_cover_sampling(node_graph, num_samples)
        return 1, node_graph, samples
    # using an integer for sim threhsold avoids lots of floating drama!
    sim_upper = sims[1]
    sim_lower = sims[0] # 707 corresponds to 0.707
    max_run = 20
    count = 0
    current_coverage = 0

    # Set sim to sim_lower to run the first iteration with sim_lower. If we
    # cannot achieve the coverage with sim_lower, then return the samples.
    sim = (sim_upper + sim_lower) / 2
    # node_graph = build_graph(data, sim / 1000, labels=labels
    cap = (2 * total_num * coverage) / num_samples
    while abs(current_coverage - coverage) > epsilon and sim_upper - sim_lower > 1:
        if count >= max_run:
            print(f"Reached max number of iterations ({max_run}). Breaking...")
            break
        count += 1

        node_graph = build_graph(data, sim / 1000, max_degree=cap, labels=labels)
        samples, rem_nodes = max_cover_sampling(node_graph, num_samples)
        current_coverage = (total_num - rem_nodes) / total_num

        if current_coverage < coverage:
            sim_upper = sim
        else:
            sim_lower = sim
        sim = (sim_upper + sim_lower) / 2
    # print(f"Converged to tau = {sim/1000}")
    return sim / 1000, node_graph, samples

def acs_sample(data_df, Ks):
    coverage = 0.9 # Coverage fixed at 0.9
    embed_data  = get_embeddings_task(data_df['sentence'])
    data_labels = data_df['label'].tolist()
    cos_sim     = cosine_similarity(embed_data)

    selected_samples = {}
    for K in tqdm(Ks, desc="Processing Ks..."):
        # print(f"Processing for K = {K}")
        _, _, selected_samples[K] = calculate_similarity_threshold(cos_sim, K, coverage, labels=data_labels)
    return selected_samples

