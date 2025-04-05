import csv
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    """
    Load the csv file in and store it in nested map.
    edgefile: The csv data have the format of 'start, end, distance'.
    graph: The nested map that stores the graph. Store the data in the format of
    {start: {end: (distance)}}. 

    For example, if the csv file is like:
    A_1, A_2, 1
    A_2, A_3, 2
    The graph should be like:
    {'A_1': {'A_2': 1}, 'A_2': {'A_3': 2}, 'A_3': {}}
    """
    graph = dict()
    with open(edgeFile, 'r') as f:
        for line in f:
            data = line.split(',') # split the data by ','
            if data[0] == 'start':
                continue
            if data[0] not in graph:
                graph[data[0]] = dict()
            if data[1] not in graph:
                graph[data[1]] = dict()

            graph[data[0]][data[1]] = float(data[2])

    """
    Init the retuen value.
    dist: A float value that stores the total distance of shortest the path.
    path: A list of string that stores the nodes of the shortest path from start to end.
    num_visited: An integer that stores the number of visited nodes.
    """
    dist = 0.0
    path = []
    num_visited = 0

    """
    Init the container for BFS.
    queue: A list of string that stores the nodes that need to be visited.
    visited: A set of string that stores the nodes that have been visited.
    parent: A map that stores the parent node of each node. 
            We use this to trace back the shortest path from start node to end node
    nodes: A dictionary that stores the nodes that are connected to the current node.
    found: A boolean value that indicates whether the end node is found.
    """
    queue = []
    queue.append(str(start))

    visited = set() # Value in set is unique
    visited.add(str(start))

    parent = {str(start): None}# parent of start node is None
    nodes = dict()
    found = False

    """
    Implement the BFS algorithm.
    1. End the loop when the queue is empty or the end node is found.
    2. In the loop:
        a. Pop the first node in the queue.
        b. Increase the number of visited nodes by 1.
        c. If the current node is the end node, set found to True and break the loop.
        d. If the current node is not the end node, add the connected nodes to the queue.
    """
    while len(queue) > 0 and found == False:
        current = queue.pop(0)
        num_visited += 1
        nodes = graph[current]
        for node in nodes:
            if node not in visited:
                visited.add(node)
                parent[node] = current
                if node == str(end):
                    found = True
                    break
                queue.append(node)

    """
    Trace the path from the end node to the start node, to get the shortest path.
    To do this, we need to:
    1. End the loop if the start node is found or the parent is None.
    2. In the loop:
        a. Insert the current node to the path list.
        b. Add the distance between the current node and the parent node to the total distance.
        c. Update the current node to the parent node.
    """
    current = str(end)
    while current != str(start):
        path.append(int(current))
        if parent[current] == None:
            break
        dist = dist + float(graph[parent[current]][current])
        current = parent[current]

    path.append(str(start))
    path.reverse()

    return path, dist, num_visited

    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')