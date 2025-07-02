import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import json
import re
import networkx as nx


################################################################################
#########################   HELPER FUNCTIONS   #################################
################################################################################


########### Three load functions to read json and jsonc files. #################
def load_jsonc(filepath):
    """Load JSONC file (JSON with comments)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove single-line comments (// ...)
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Remove trailing commas before closing brackets/braces
    content = re.sub(r',\s*([}\]])', r'\1', content)
    
    return json.loads(content)


def load_json(filepath):
    """Load regular JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_any(filepath):
    """Load either JSON or JSONC file based on extension"""
    if filepath.endswith('.jsonc'):
        return load_jsonc(filepath)
    else:
        return load_json(filepath)


########### visualize the hierarchy ##########################
def describe_hierarchy(model):
    G          = model.shared_data['region_id_graph']
    id_to_name = model.shared_data['id_to_name']

    depths = {n: nx.shortest_path_length(G, 1, n) for n in G.nodes}

    for n in nx.dfs_preorder_nodes(G, 1):
        indent = '  ' * depths[n]
        print(f"{indent}{id_to_name[n]} ({n})")


def describe_data(model):
    G               = model.shared_data['region_id_graph']
    data            = model.shared_data['data']
    id_to_name      = model.shared_data['id_to_name']
    
    for n in nx.dfs_postorder_nodes(G, 1):
        cnt = data['location_id'].eq(n).sum() + sum(G.nodes[c].get('cnt', 0) for c in G.successors(n))
        G.nodes[n]['cnt'] = int(cnt)
        G.nodes[n]['depth'] = nx.shortest_path_length(G, 1, n)
        
    for n in nx.dfs_preorder_nodes(G, 1):
        if G.nodes[n]['cnt'] > 0:
            print('  '*G.nodes[n]['depth'] + id_to_name[n] + f' ({n}): ', G.nodes[n]['cnt'])


################################################################################
#########################   MAIN FUNCTIONS   ###################################
################################################################################


def initiliaze_pipeline(filepath, data_type='p', verbose=False):


    ############## 1. Load inputs data ##########################################
    input_data      = pd.read_csv(f'{filepath}/input_data.csv')
    output_template = pd.read_csv(f'{filepath}/output_template.csv')
    parameters      = load_any(f'{filepath}/parameters.jsonc')    
    hierarchy       = load_any(f'{filepath}/hierarchy.json')       
    nodes_to_fit    = load_any(f'{filepath}/nodes_to_fit.json') 

    # filter input_data by data_type
    unfiltered_input_data = input_data.copy()
    input_data = input_data[input_data['data_type'] == data_type]

    # create region_id_graph with hierarchy
    nodes = hierarchy['nodes']
    name_to_id = {} # NOTE: this can't handle duplicate names
    id_to_name = {}

    region_id_graph = nx.DiGraph()

    for node in nodes:
        name_to_id[node[0]] = node[1]['location_id']
        id_to_name[node[1]['location_id']] = node[0]

        # add nodes with location_id as the key
        region_id_graph.add_node(
                                node[1]['location_id'],           # location_id is the node key
                                level = node[1]['level'],
                                parent_id = node[1]['parent_id'],
                                name = node[0]
                                )

        # add edges between nodes (ignore root node)
        my_id = node[1]['location_id']
        parent_id = node[1]['parent_id']
        if my_id != parent_id: # ignores root node
            region_id_graph.add_edge(parent_id, my_id)


    # since the graph is a tree, the number of nodes should be equal to the number of edges + 1
    assert region_id_graph.number_of_nodes() == region_id_graph.number_of_edges() + 1, \
        "number of nodes should be equal to the number of edges + 1"
    

    ############## 2. Initialize pm.Model() and shared_data #####################
    pm_model = pm.Model()

    with pm_model: 
        pm_model.shared_data = {     # NOTE: this is what used to be "vars" from class ModelVars
            "unfiltered_input_data"  : unfiltered_input_data,
            "data_type"              : data_type,
            "data"                   : input_data,
            "region_id_graph"        : region_id_graph,
            "id_to_name"             : id_to_name,
            "name_to_id"             : name_to_id,
            "output_template"        : output_template,
            "ages"                   : np.array(parameters['ages'], dtype=np.float64),
            "age_weights"            : np.array(parameters['age_weights'], dtype=np.float64),
            "params_of_data_type"    : parameters[data_type],
        }


    ############## 3. Print summary ##############################################
    if verbose:
        print(f'number of rows: {len(input_data)}')
        print(f'number of unique location_id: {input_data["location_id"].nunique()}')
        print(f"number of nodes: {region_id_graph.number_of_nodes()}") 
        print(f"number of edges: {region_id_graph.number_of_edges()}")
        
    return pm_model