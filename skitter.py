import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_network(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                node1, node2 = line.strip().split()
                G.add_edge(node1, node2)
    return G

def analyze_network(G):
    # 平均度
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    
    # 密度
    density = nx.density(G)
    
    # 聚集系数
    clustering_coeff = nx.average_clustering(G)
    
    # 度分布
    degree_hist = nx.degree_histogram(G)
    degree_range = range(len(degree_hist))
    
    # 绘制度分布
    plt.loglog(degree_range, degree_hist, 'b-', marker='o')
    plt.title("Degree Distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Degree")
    plt.show()
    
    return avg_degree, density, clustering_coeff

def power_law_fit(degrees):
    # 计算度分布并拟合幂律分布
    counts, bins = np.histogram(degrees, bins='auto', density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    
    # 幂律分布函数
    def power_law(x, a, b):
        return x**-a * np.exp(-b / x)
    
    params, cov = curve_fit(power_law, centers, counts)
    
    # 绘制拟合结果
    plt.plot(centers, counts, 'b-', marker='o', label='Data')
    plt.plot(centers, power_law(centers, *params), 'r-', label='Fit: a=%5.3f, b=%5.3f' % tuple(params))
    plt.legend()
    plt.xlabel('Degree')
    plt.ylabel('P(Degree)')
    plt.loglog()
    plt.show()
    
    return params

def generate_ba_model(n, m):
    # 使用 Barabási-Albert 模型生成网络
    G_ba = nx.barabasi_albert_graph(n, m)
    return G_ba

def estimate_average_shortest_path_length(G):
    try:
        # 对于大型网络，这可能非常耗时
        avg_path_length = nx.average_shortest_path_length(G)
    except Exception as e:
        print(f"Error calculating average shortest path length: {e}")
        avg_path_length = None
    return avg_path_length

# 假设文件路径
file_path = 'as-skitter.txt'
G = load_network(file_path)

# 分析网络
avg_degree, density, clustering_coeff = analyze_network(G)
print(f"Average Degree: {avg_degree}, Density: {density}, Clustering Coefficient: {clustering_coeff}")

# 生成 BA 模型网络并分析
n = len(G.nodes())
m = int(avg_degree / 2) # 假设每个新增节点附加到m个现有节点
G_ba = generate_ba_model(n, m)
_, density_ba, clustering_coeff_ba = analyze_network(G_ba)

# 计算平均最短路径长度（可能非常慢）
# avg_path_length = estimate_average_shortest_path_length(G)
# print(f"Average Shortest Path Length: {avg_path_length}")

# 使用替代方案加速计算，如利用并行计算、近似算法或切换到更高效的图处理库，例如 graph-tool 或 igraph。

