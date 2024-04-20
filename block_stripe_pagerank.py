import numpy as np 
import pickle as pkl
data_path = 'Data.txt'

LINK_MATRIX_PATH_PREFIX = ".//block_stripe//M//matrix_"
LINK_MATRIX_PATH_SUFFIX = ".pkl"
R_VECTOR_PATH_PREFIX = ".//block_stripe//vector//r_vector_"
R_VECTOR_PATH_SUFFIX = ".pkl"
R_NEW_VECTOR_PATH_PREFIX = ".//block_stripe//new_vector//r_vector_"
R_NEW_VECTOR_PATH_SUFFIX = ".pkl"
RESULT_PATH = ".//block_stripe//result//result.txt"
OUTPUT_NUM = 100  # top 100 nodes

max_node_index=8297  #the maximum number of nodes in the graph
belta = 0.85  #the probability of following the link
epsilon = 1e-8   #the threshold of the difference between the old and new pagerank
Num=8298  #the number of nodes in the graph

group_num=20 #分组数量

def get_group_id(node_id):
    return node_id // (Num //group_num+1)

def get_group_size():
    return Num//group_num+1

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
    link_matrix_groups=[[[i,0,[]] for i in range(max_node_index+1)] for j in range(group_num)]
    #分块
    for line in link_matrix:
        for dest in line[2]:
            group_id=get_group_id(dest)
            link_matrix_groups[group_id][line[0]][1]=line[1]  #注意这里的degree是一样的
            link_matrix_groups[group_id][line[0]][2].append(dest)
    #把每个group中的出度为0的节点删除
    for i in range(group_num):
        link_matrix_groups[i]=[link_matrix_groups[i][j] for j in range(max_node_index) if link_matrix_groups[i][j][1]>0]
    #保存到文件
    for i in range(group_num):
        with open(LINK_MATRIX_PATH_PREFIX+str(i)+LINK_MATRIX_PATH_SUFFIX,'wb') as f:
            for line in link_matrix_groups[i]:
                pkl.dump(line,f)


def compute_rnew(flag):
    error=0.0
    for i in range(group_num):
        #initialize r_new_stripe
        r_new_stripe =np.ones(get_group_size())*1.0*(1-belta) / Num
        #读取r_old_stripe
        if flag==0:
            with open(R_VECTOR_PATH_PREFIX+str(i)+R_VECTOR_PATH_SUFFIX,'rb') as f:
                r_old_stripe=pkl.load(f)
        else:
            with open(R_NEW_VECTOR_PATH_PREFIX+str(i)+R_NEW_VECTOR_PATH_SUFFIX,'rb') as f:
                r_old_stripe=pkl.load(f)
        with open(LINK_MATRIX_PATH_PREFIX+str(i)+LINK_MATRIX_PATH_SUFFIX,'rb') as fl:
            while True:
                try:
                    # 逐行读取
                    row_data = pkl.load(fl)
                    # print(row_data)
                    src, degree, dest = row_data
                    #读出r_old[src]
                    r_index=get_group_id(src)
                    if flag==0:
                        with open(R_VECTOR_PATH_PREFIX+str(r_index)+R_VECTOR_PATH_SUFFIX,'rb') as f_r:
                            r_tmp_stripe=pkl.load(f_r)
                    else:
                        with open(R_NEW_VECTOR_PATH_PREFIX+str(r_index)+R_NEW_VECTOR_PATH_SUFFIX,'rb') as f_r:
                            r_tmp_stripe=pkl.load(f_r)
                    for k in range(len(dest)):
                        dest_index=dest[k]%get_group_size()
                        r_new_stripe[dest_index] += belta * r_tmp_stripe[src%get_group_size()] / degree
                except EOFError:
                    break  # 如果到达文件末尾，跳出循环
        #计算误差
        error+=np.sum(np.abs(r_new_stripe-r_old_stripe))
        #保存r_new至磁盘上
        if flag==0:
            with open(R_NEW_VECTOR_PATH_PREFIX+str(i)+R_NEW_VECTOR_PATH_SUFFIX,'wb') as fr:
                pkl.dump(r_new_stripe,fr)
        else:
            with open(R_VECTOR_PATH_PREFIX+str(i)+R_VECTOR_PATH_SUFFIX,'wb') as fr:
                pkl.dump(r_new_stripe,fr)
    return error

def block_stripe_pagerank():
    #initialize the r_old
    #分块 储存
    for i in range(group_num):
        r_stripe =np.ones(get_group_size())*1.0 / Num
        with open(R_VECTOR_PATH_PREFIX+str(i)+R_VECTOR_PATH_SUFFIX,'wb') as f:
            pkl.dump(r_stripe,f)
    #compute the r_new
    flag=0
    while True:
        error=compute_rnew(flag%2)
        flag+=1
        print("error:",error)
        if error<epsilon:
            break

def print_result():
    #降序排列
    results=[]
    #先从磁盘读取r_stripe
    for i in range(group_num):
        r_stripe=pkl.load(open(R_VECTOR_PATH_PREFIX+str(i)+R_VECTOR_PATH_SUFFIX,'rb'))
        #保存top100至results
        index = np.argsort(-r_stripe)
        for j in range(OUTPUT_NUM):
            results.append((index[j]+i*get_group_size(),r_stripe[index[j]]))
    #top100排序
    results.sort(key=lambda x:x[1],reverse=True)
    #output the result
    with open(RESULT_PATH, "w") as f:
        for i in range(OUTPUT_NUM):
            f.write(f"{results[i][0]} {results[i][1]}\n")
            
    

if __name__ == '__main__':
    print("##############Block and Stripe PageRank begin##############")
    load_data()
    print("Data loaded")
    print("Begin to compute the pagerank")
    block_stripe_pagerank()
    print_result()




