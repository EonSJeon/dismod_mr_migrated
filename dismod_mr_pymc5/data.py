import pandas as pd
import networkx as nx
import json
import os
import re
import numpy as np
<<<<<<< HEAD
=======
import dismod_mr_pymc5
>>>>>>> 5d6a5ef310d1fa79ee006a621d7ea78e739a2334

class MRModel:
    """
    MRModel is a container and manager for all data, parameters, and structures needed to build and fit a hierarchical Bayesian disease model using PyMC.

    Attributes:
        input_data (pd.DataFrame): Epidemiological input data for model fitting.
        output_template (pd.DataFrame): Template for model output structure.
        parameters (dict): Model parameters, priors, and configuration by rate type.
        hierarchy (nx.DiGraph): Directed graph representing the area or region hierarchy.
        nodes_to_fit (list): List of nodes (areas) in the hierarchy to be fit.
        vars (dict): References to PyMC variables created during model construction.
        model (pm.Model or None): The PyMC model object, set after model construction.
        model_settings (dict): Additional settings for model configuration and fitting.

    Methods:
        (To be implemented) setup_model, fit, predict, save, load, etc.
    """


    def __init__(self):
        """
        Initialize an empty MRModel instance with default data structures.

        Sets up empty DataFrames for input and output, initializes parameters and hierarchy,
        and prepares containers for PyMC variables and model settings. No model is built at this stage.
        """
        self.input_data = pd.DataFrame(
            columns=[
                'data_type', 'value', 'area', 'sex',
                'age_start', 'age_end', 'year_start', 'year_end',
                'standard_error', 'effective_sample_size',
                'lower_ci', 'upper_ci', 'age_weights'
            ]
        )

        self.output_template = pd.DataFrame(columns=['data_type','area','sex','year','pop'])

        self.parameters = dict(i={}, p={}, r={}, f={}, rr={}, X={}, pf={}, ages=list(range(101))) 

        self.hierarchy = nx.DiGraph()
        self.hierarchy.add_node('Global')
        self.nodes_to_fit = list(self.hierarchy.nodes())

        self.vars = {}
        self.model = None
        self.model_settings = {}


    def get_data(self, data_type: str) -> pd.DataFrame:
        """
        Get the input data for a specific data type.
        
        :Parameters:
          - `data_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf', 'm', 'X', or 'csmr'

        """
        return self.input_data[self.input_data['data_type'] == data_type]
    

    def keep(self, areas: list = ['Global'], sexes: list = ['Male','Female','Both'], start_year: int = -np.inf, end_year: int = np.inf):
        """
        Filter the input data to only include the specified areas, sexes, and year range.
        """

        # Filter by area
        if 'Global' not in areas:
            self.hierarchy.remove_node('Global')
            for area in areas:
                self.hierarchy.add_edge('Global', area)
            self.hierarchy = nx.bfs_tree(self.hierarchy, 'Global')
            relevant = self.input_data['area'].isin(self.hierarchy.nodes()) | (self.input_data['area']=='Global')
            self.input_data = self.input_data.loc[relevant]
            self.nodes_to_fit = list(set(self.hierarchy.nodes()) & set(self.nodes_to_fit))

        # Filter by sex
        self.input_data = self.input_data[self.input_data['sex'].isin(sexes)]

        # Filter by year
<<<<<<< HEAD
        self.input_data = self.input_data[self.input_data['year_id'] >= start_year]
        self.input_data = self.input_data[self.input_data['year_id'] <= end_year]
=======
        self.input_data = self.input_data[self.input_data['year_end'] >= start_year]
        self.input_data = self.input_data[self.input_data['year_start'] <= end_year]
>>>>>>> 5d6a5ef310d1fa79ee006a621d7ea78e739a2334

        print(f'kept {len(self.input_data)} rows of data')


    @staticmethod
    def load(path: str) -> 'MRModel':
        
        def load_jsonc(fp):
            txt = open(fp, encoding='utf-8').read()
            no_comments = re.sub(r'//.*?$|/\*.*?\*/', '', txt, flags=re.MULTILINE|re.DOTALL)
            clean = re.sub(r',\s*([}\]])', r'\1', no_comments)
            return json.loads(clean)
        
        def load_any(name):
            for ext, loader in (('.jsonc', load_jsonc), ('.json', lambda fp: json.load(open(fp, encoding='utf-8')))):
                fp = os.path.join(path, f"{name}{ext}")
                if os.path.isfile(fp):
                    return loader(fp)
            raise FileNotFoundError(f"No {name}.json(c) in {path}")
        
        d = MRModel()
        d.input_data = pd.read_csv(os.path.join(path, 'input_data.csv'))
        d.output_template = pd.read_csv(os.path.join(path, 'output_template.csv'))
        d.parameters = load_any('parameters')

        hier = load_any('hierarchy')
        d.hierarchy = nx.DiGraph()
        
        # Add nodes with their metadata attributes
        for location_name, location_info in hier['nodes']:
            d.hierarchy.add_node(location_name, **location_info)
        
        # Filter out self-loops and problematic edges
        valid_edges = []
        for parent_name, child_name in hier['edges']:
            if parent_name != child_name:  # Skip self-loops
                valid_edges.append((parent_name, child_name))
            else:
                print(f"Warning: Skipping self-loop edge: {parent_name} -> {child_name}")
        
        d.hierarchy.add_edges_from(valid_edges)
        d.nodes_to_fit = load_any('nodes_to_fit')
        
        print(f"Loaded hierarchy with {len(d.hierarchy.nodes())} nodes and {len(valid_edges)} edges")
        return d
    

    def describe(self, data_type: str):
        G = self.hierarchy
        df = self.get_data(data_type)
        for n in nx.dfs_postorder_nodes(G, 'Global'):
            cnt = df['area'].eq(n).sum() + sum(G.nodes[c].get('cnt', 0) for c in G.successors(n))
            G.nodes[n]['cnt'] = int(cnt)
            G.nodes[n]['depth'] = nx.shortest_path_length(G, 'Global', n)
            
        for n in nx.dfs_preorder_nodes(G, 'Global'):
            if G.nodes[n]['cnt'] > 0:
                print('  '*G.nodes[n]['depth'] + n, G.nodes[n]['cnt'])

load = MRModel.load