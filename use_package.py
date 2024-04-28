import sys
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt

DATA_PATH = ".\\Data.txt"
PRINT_NUM = 100

def pagerank():
    f = open(DATA_PATH, 'rb')
    p = f.readlines()

    G = nx.DiGraph()
    for i in range(0, len(p)):
        nodes_split = str(p[i], "utf-8").split()
        fm = nodes_split[0]
        to = nodes_split[1]
        G.add_edge(fm, to)
    print("load data has finished..")

    pr = nx.pagerank(G, alpha=0.85,tol=1e-8)
    pr_dic = {}
    for node, pageRankValue in pr.items():
        pr_dic[node] = pageRankValue
    print("calculate pagerank has finished..")
    #累加所有节点权值

    data_list = [{k: v} for k, v in pr_dic.items()]
    f = lambda x: list(x.values())[0]
    sorted_dic = sorted(data_list, key=f, reverse=True)
    with open("result.txt", "w") as f:
        for i in range(0, PRINT_NUM):
            #键  值
            f.write(str(list(sorted_dic[i].keys())[0]) + " " + str(list(sorted_dic[i].values())[0] ) + "\n")
    


if __name__ == "__main__":
    pagerank()