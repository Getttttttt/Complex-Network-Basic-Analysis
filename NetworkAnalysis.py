import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import logging

class NetworkAnalysis:
    def __init__(self, file_path, directed=False):
        self.file_path = file_path
        self.directed = directed
        self.G = None
        self.network_name = file_path.split('/')[-1].split('.')[0]
        
        # 设置日志记录
        logging.basicConfig(filename=f"{self.network_name}_error.log", level=logging.ERROR)
        
    def load_network(self):
        try:
            if self.file_path.endswith('.txt'):
                self.G = nx.read_edgelist(self.file_path, create_using=nx.DiGraph() if self.directed else nx.Graph())
            elif self.file_path.endswith('.csv'):
                edges = pd.read_csv(self.file_path)
                self.G = nx.from_pandas_edgelist(edges, 'node_1', 'node_2', create_using=nx.DiGraph() if self.directed else nx.Graph())
            nx.write_pajek(self.G, f"{self.network_name}.net")
        except Exception as e:
            logging.error(f"Error loading network: {e}")
    
    def analyze_network(self):
        try:
            degrees = [d for n, d in self.G.degree()]
            avg_degree = np.mean(degrees)
            density = nx.density(self.G)
            clustering_coeff = nx.average_clustering(self.G)
            with open(f"{self.network_name}_indicators.txt", "w") as f:
                f.write(f"Average Degree: {avg_degree}\nDensity: {density}\nClustering Coefficient: {clustering_coeff}\n")
            return avg_degree, density, clustering_coeff
        except Exception as e:
            logging.error(f"Error analyzing network: {e}")
    
    def degree_distribution(self):
        try:
            degrees = [d for n, d in self.G.degree()]
            plt.hist(degrees, bins='auto')
            plt.title("Degree Distribution")
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.savefig(f"{self.network_name}_Degree_Distribution.png")
            plt.close()
        except Exception as e:
            logging.error(f"Error generating degree distribution plot: {e}")
    
    def estimate_average_shortest_path_length(self):
        try:
            if nx.is_connected(self.G if not self.directed else self.G.to_undirected()):
                avg_path_length = nx.average_shortest_path_length(self.G)
                with open(f"{self.network_name}_indicators.txt", "a") as f:
                    f.write(f"Average Shortest Path Length: {avg_path_length}\n")
                return avg_path_length
            else:
                return None
        except Exception as e:
            logging.error(f"Error calculating average shortest path length: {e}")
    

network_paths = ['HR_edges.csv', 'HU_edges.csv', 'RO_edges.csv']
for path in network_paths:
    na = NetworkAnalysis(f"deezer_clean_data/{path}", directed=True)
    na.load_network()
    na.analyze_network()
    na.degree_distribution()
    na.estimate_average_shortest_path_length()

na = NetworkAnalysis("as-skitter.txt", directed=False)
na.load_network()
na.analyze_network()
na.degree_distribution()
na.estimate_average_shortest_path_length()