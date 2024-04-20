import numpy as np 
import pickle as pkl
data_path = 'Data.txt'

LINK_MATRIX_PATH = ".//basic_block//M//link_matrix.pkl"
R_VECTOR_PATH = ".//basic_block//vector//r_vector.pkl"
RESULT_PATH = ".//basic_block//result.txt"
OUTPUT_NUM = 20  #the number of nodes to output

max_node_index=8297  #the maximum number of nodes in the graph
belta = 0.85  #the probability of following the link
epsilon = 1e-8   #the threshold of the difference between the old and new pagerank
Num=8298  #the number of nodes in the graph

def load_data():
    # Load the data from the file
    global max_node_index
    global Num
    global nodes_set
    link_matrix = [[i,0,[]] for i in range(max_node_index+1)]  #link matrix   src degree dest[]
    max_node_index=0
    with open(data_path) as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.split()
            fr, to = int(split_line[0]), int(split_line[1])
            link_matrix[fr][1]+=1   #degree
            link_matrix[fr][2].append(to)  #dest
            max_node_index = max(max_node_index,fr,to)
    Num=max_node_index+1
    #把link_matrix中出度为0的节点删除
    link_matrix=[link_matrix[i] for i in range(max_node_index+1) if link_matrix[i][1]>0]
    with open(LINK_MATRIX_PATH, "wb") as f:
        for row in link_matrix:
            pkl.dump(row, f)

def compute_rnew(r_old):
    r_new = np.zeros(Num)*1.0
    with open(LINK_MATRIX_PATH, 'rb') as f:
        while True:
            try:
                # 逐行读取
                row_data = pkl.load(f)
                # print(row_data)
                src, degree, dest = row_data
                for i in range(degree):
                    r_new[dest[i]] += belta * r_old[src] / degree
            except EOFError:
                break  # 如果到达文件末尾，跳出循环
    #re-insert the leaked
    leaked=np.ones(Num)*(1-belta)/Num
    r_new+=leaked
    return r_new

def basic_pagerank():
    #initialize the r_old
    r_old = np.ones(Num)*1.0 / Num
    #compute the r_new
    r_new = compute_rnew(r_old)
    #iteration
    while np.sum(np.abs(r_new - r_old)) > epsilon:
        print(np.sum(np.abs(r_new - r_old)))
        r_old = r_new
        r_new = compute_rnew(r_old)
    #保存结果
    pkl.dump(r_new,open(R_VECTOR_PATH,'wb'))
    return r_new

def print_result(r_new):
    #降序排列
    index = np.argsort(-r_new)
    #output the result
    for i in range(100):
        print("The No.%d node is %d, and the pagerank is %.10f" % (i, index[i], r_new[index[i]]))
    with open(RESULT_PATH, "w") as f:
        for i in range(100):
            f.write(f"{index[i]} {r_new[index[i]]}\n")

if __name__ == '__main__':
    load_data()
    r_new = basic_pagerank()
    print_result(r_new)




