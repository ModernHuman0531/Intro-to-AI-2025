import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    """
    Load the csv file in and store it in nested map.
    edgefile: The csv data that have the format of 'start' ,'end', 'distance'.
    graph: The nested map that store tje graph. Store the data in the format of
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
    Init the return value.
    dist: The float number that store the total distance of the shortest path.
    path: The list of the string that store the nodes of the shortest path from start to end
    num_visited: The integer that store the number of visited nodes.
    """
    dist = 0.0
    path = []
    num_visited = 0

    """
    Init the container for DFS.
    stack: The list of the string that store the nodes that need to be visited.
    parent: The map that store the parent node of each node.
            We use this to trace back the shortest path from start node to end node.
    visited: The set of the string that store the nodes that have been visited.
    nodes: The dicitionary that store the nodes that are connected to the current node.
    found: The boolean value that indicate whetehr the end node is found or not.
    """
    stack = []
    stack.append(str(start))

    visited = set()
    visited.add(str(start))

    parent = {str(start) : None} # parent of the start node is None
    nodes = dict()
    found = False

    """
    Implemented the DFS alopgrithm.
    1.End the loop when stack is empty or the end node is foupath.insert(current)
    2. In the loop:
        a. Pop the current node from the stack.
        b. Check if the current node is visited or not.
        c. If the current node is the end node, set the found to True and break the loop.
        d. Increase the number of visited nodes by 1.
        e. If not, add the connected nodes into the stack.
    """
    while len(stack) > 0 and found == False:
        current = stack.pop()
        num_visited += 1
        nodes = graph[current]
        for node in nodes:
            if node not in visited:
                visited.add(node)
                parent[node] = current
                if node == str(end):
                    found = True
                    break
                stack.append(node)
        
    """
    Trace back the shortest path from the end node to the start node to achieve the distace and path.
    To do this, we need to:
    1. End the loop when we find the start node or the parent of thwe current node is None.
    2. In the loop:
        a. Add the distance vector between the current node and the parent node to the total distance.
        b. Insert the current node to the path list.
        c. Update the current node to the parent node.
    """
    current = str(end)
    while current != str(start) and parent[current] != None:
        path.append(int(current))
        if parent[current] == None:
            break
        dist = dist + graph[parent[current]][current]
        current = parent[current]

    path.append(str(start))
    path.reverse()
    
    return path, dist, num_visited

    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
