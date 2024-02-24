import torch
def pdist(a,dim=2, p=2):
    dist_matrix = torch.norm(a[:, None]-a, dim, p)
    return dist_matrix
def floyd_warshall_algorithm(adj_matrix):
    n = adj_matrix.size(0)
    distance_matrix = adj_matrix.clone()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

    return distance_matrix

def abandonOverSizeEdge(max_dis,dis_matrix):
	connect_matrix = dis_matrix.clone()
	for i in range(dis_matrix.shape[0]):
		for j in range(dis_matrix.shape[1]):
			if dis_matrix[i][j]>max_dis:
				connect_matrix[i][j]=0
			else:
				connect_matrix[i][j]=1
	return connect_matrix
def dfs(graph, visited, current, k,label):
    visited[current] = True
    label[current]=k 
    index=0
    for neighbor in graph[current]:
        if not visited[index]:
            if neighbor==1:
                dfs(graph, visited, index, k,label)
        index+=1
def all_connected_components(graph):
    n = graph.size(0)
    visited = torch.zeros(n, dtype=torch.bool)
    components = []
    label=torch.zeros(n,dtype=torch.int)
    k=0
    for i in range(n):
        if not visited[i]:
            
            dfs(graph, visited, i, k,label)
            k+=1

    return label
	
def getCluster(points,maxdis):
	points=torch.Tensor(points)
	dis=pdist(points)
	dis=floyd_warshall_algorithm(dis)
	dis=abandonOverSizeEdge(maxdis,dis)
	return all_connected_components(dis).numpy()