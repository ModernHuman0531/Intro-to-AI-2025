import csv
import queue
edgeFile = 'edges.csv'


def ucs(start, end):
    # Begin your code (Part 3)
    """
    Load the csv file in a nd store it in nested map.
    edgeFile: The csv data that have the format of 'start', 'end', 'distance'.
    graph: The nested map that store the graph. Store the data in the format of
    {start: {end: (distance)}}.
    """
    graph = dict()
    with open(edgeFile, 'r') as f:
        for line in f:
            data = line.split(',')
            if data[0] == 'start':
                continue
            if data[0] not in graph:
                graph[data[0]] = dict()
            if data[1] not in graph:
                graph[data[1]] = dict()
            
            graph[data[0]][data[1]] = float(data[2])

    """
    Init the return value
    Path: The list that store the path from start node to end node
    Dist: The float that store the total distance of the path
    Num_visited: The int that store the number of visited nodes
    """
    path = []
    dis = 0.0
    num_visited = 0

    """
    Init the container for the UCS algorithm
    heap: The list that store the nodes that need to be visited. The format of the node is
            (distance, node)
    visited: The set that store the nodes that have been visited
    parent: The map that store the parent node of each node. We use this to trace back the shortest path
    found: The boolean value that indicate whether the end node is found or not
    """
    # Priority queue can arrange the nodes in the order of distance.
    heap = queue.PriorityQueue()
    heap.put((0.0, str(start)))

    visited = set()
    visited.add(str(start))

    parent = {str(start): (None, 0.0)} # parent of the start node is None, and the distance is 0.0
    found = False
    # End your code (Part 3)
    """
    Implemented the UCS algorithm
    1. End the loop when the heap is empty or the node is found.
    2. In the loop:
        a.  Pop the node with the smallest distance from the heap
        b. If the node is the end node or the queue is empty, end the loop
        c. Else add it to the visited set.
        d. Get the connected nodes of the current node
        e. For each node, calculate the distance from start node to the node.
        f. If the node is not visited or we found the shorter path, update the parent node and add it to the heap.

    For example:
    A --1-- B --2-- C
    |       |       |
    4       5       1
    |       |       |
    C ----1---- D
    The reason why we need to update the parent node is that we need to trace back the shortest path.
    If we just see through a -> c, there are two paths: a -> b -> c and a -> c. If we found the shorter path
    ,we first find a->c, so we set the parent of c is a. Then we find a->b->c, we need to update the parent of c
    to b. So we can trace back the shortest path.
    """
    while heap.empty() == 0 and found == False:
        current = heap.get()
        num_visited += 1
        if str(current[1]) == str(end):
            found= True
            break
        nodes = graph[current[1]]
        for node in nodes:
            # Current[0] is the distance from start node to current node,
            # graph[current[1]][node] is the distance from current node to the connected node 
            start_to_node = current[0] + graph[current[1]][node]
            if node not in visited or start_to_node < parent[node][1]: # parent[node][1] is the distance already found to reach the current node
                visited.add(node)
                parent[node] = (current[1], start_to_node)
                heap.put((start_to_node, str(node)))

    """
    Traceback the shortest path
    1. End the loop when the parentnode is None or the current node is start node
    2. In the loop:
        a. Add the distance to the total distance
        b. Insert the current node to the path list
        c. Update the current node to the parent node
    """
    current = str(end)
    while current != str(start) and parent[current][0] != None:
        dis = dis + float(graph[parent[current][0]][current])
        path.append(int(current))
        current = parent[current][0]

    path.append(str(start))
    path.reverse()

    # The error of dis might 
    return path, dis, num_visited


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
