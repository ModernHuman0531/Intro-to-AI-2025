import csv
import queue
edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'


def astar(start, end):
    # Begin your code (Part 4)
    """
    Load the csv file in and store it in nested map.
    edgefile; Is a csv file that stores the data it the format of 'start', 'end', 'distance'
    graph: A nested map that store the graph. Store the data in the format of
    {start: {end: (distance)}}
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
    Load the heuristic csv file and store it in a map.
    heuristic: Is a csv file that stores the data in the format of 'node', 'distance of node to destination 1', 'distance of node to destination 2', 'distance of node to destination 3'
    graph: A map that store the heuristic. Store the data in the format of:
    {node: Destination}
    """
    straight_dis = dict()
    with open(heuristicFile, 'r') as f:
        for line in f:
            data = line.split(',') # split the data by ','
            # data[3] = 8461.89\n
            data[3] = data[3].split('\n')[0] # data[3]=8461.89, data[3].split('\n)[1]='\n'
            if data[0] == 'node':
                # Identify what the destination is
                for index, element in enumerate(data):# Enumerate relate the data with the index.
                    if element == str(end):# Chooses which is the destination
                        case = index
                        break
                continue
            straight_dis[data[0]] = float(data[case])
    
    """
    Init the return value
    dist: a float number that stoe the total distance of the shortest path
    path: a list of string that store the shortest path from start to end
    num_visited: an integer that store the number of visited nodes
    """
    dist = 0.0
    path = []
    num_visited = 0

    """
    Init the container for A* algorithm
    heap: a priority queue that store the nodes that need to be visisted. The format of of the node is
            (weight, node)
    visited" a set of nodes that already visited
    parent: a map that store trhe partent of each nodes, we use this to trace back the shortest path
            , the format of the node is {node:(parent, distance)}
    nodes: a dictionary that store the nodes that are connected to the current node
    found: A boolean that indicate whether the end is found or not.
    """
    heap = queue.PriorityQueue()
    heap.put((0+straight_dis[str(start)], str(start)))

    visited = set()
    visited.add(str(start))

    parent = dict()
    parent[str(start)] = (None, 0.0) # The parent of the start node is None, and the straight_line distance is 0.0
    nodes = dict()
    found = False
    # End your code (Part 4)

    """
    Impleemented the A* algorithm
    weight: g() + h()
        g(): The distance from the start node to the current node
        h(): The straight line distance from the current node to the end node.
    1. End the loop when the heap is  empty or found is True
    2. In the loop:
        a. Pop the the first node in the heap(the smallest distance)
        b. If the node is the end node, end the loop
        c. Find the connected nodes of the current node
        d. For each node, calculate the weight of the node
        e. If the node is not visited or the weight is smaller than the previous one, update the parent node and add it to the heap
            - Add to the visited set
            - Record the parent node and the distance in the dict()
            - Add the connected nodes to the heap
    """
    while heap.empty() == False and found == False:
        current = heap.get()
        num_visited += 1
        if current[1] == str(end):
            found = True
            break
        nodes = graph[current[1]]
        for node in nodes:
            # Distance from start to current node + distance from current node to the node +  straight_line distance from the node to the end
            weight = (current[0] - straight_dis[current[1]]) + graph[current[1]][node] + straight_dis[node]
            if node not in visited or weight < parent[node][1]:
                visited.add(node)
                parent[node] = (current[1], weight)
                heap.put((weight, node))

    """
    Trace back the shortest path from the end node to the start node
    1. End the loop when the start node is found or the parent is None.
    2. In the loop:
        a. Insert the node in the path,
        b. Add the distance between the current node and the parent node to the total distance
        c.Replace the current node with the parent node
    """
    current = str(end)
    while current != str(start):
        if parent[current] == None:
            break
        path.append(int(current))
        dist = dist + float(graph[parent[current][0]][current])
        current = parent[current][0]
    
    path.append(str(start))
    path.reverse()
    return path, dist, num_visited

if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
