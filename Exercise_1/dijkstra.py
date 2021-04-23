import numpy as np

class Dijkstra_path:
    def __init__(self, grid, source, dest):
        """
        :param grid: (np.array) nxm
        :param source: (np.array) 2
        :param dest: (np.array) 2

        """
        self.size = grid.shape
        graph = self.make_adj(grid.flatten())

        self.path = self.dijkstra(graph, tuple(source), tuple(dest))

    def make_adj(self,pixel_list):
        """
        TODO: add dosstring

        """
        num_pixels = self.size[0] * self.size[1]
        adj_matrix = np.zeros((num_pixels, num_pixels))
        neigh_str = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        neigh_dia = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        obstacle_list = []
        pixel_id_list = []

        # This code needs to be copied into the main dijkstra's algorithm function
        pixel_dict = {}
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                pixel_dict[(i, j)] = pixel_list[i * self.size[1] + j]
                pixel_id_list.append((i, j))

                if pixel_dict[(i, j)] == 3:
                    obstacle_list.append((i, j))

        for i in range(num_pixels):
            for j in range(num_pixels):
                if i == j:
                    adj_matrix[i][j] = 0
                elif self.is_str_neigh(i, j, pixel_id_list, neigh_str):
                    adj_matrix[i][j] = .5
                elif self.is_dia_neigh(i, j, pixel_id_list, neigh_dia):
                    adj_matrix[i][j] = 1
                else:
                    adj_matrix[i][j] = np.Inf

        for k in obstacle_list:
            obs_index = pixel_id_list.index(k)
            # print(obs_index)
            adj_matrix[obs_index][:] = 10
            for h in range(num_pixels):
                adj_matrix[h][obs_index] = 10

        t_graph = {}
        for i in range(num_pixels):
            t_graph[pixel_id_list[i]] = {}
            for j in range(num_pixels):
                if adj_matrix[i][j] != np.Inf:
                    t_graph[pixel_id_list[i]][pixel_id_list[j]] = adj_matrix[i][j]

        return t_graph

    def is_str_neigh(self,i, j, p_list, neigh):
        """
        TODO: add dosstring

        """
        row_cmp = p_list[i]
        col_cmp = p_list[j]

        neigh_list = [(row_cmp[0] + x[0], row_cmp[1] + x[1]) for x in neigh]
        if col_cmp in neigh_list:
            return True
        else:
            return False

    def is_dia_neigh(self,i, j, p_list, neigh):
        row_cmp = p_list[i]
        col_cmp = p_list[j]

        neigh_list = [(row_cmp[0] + x[0], row_cmp[1] + x[1]) for x in neigh]
        if col_cmp in neigh_list:
            return True
        else:
            return False

    def dijkstra(self, graph, start, goal):
        """
        TODO: add dosstring

        """
        shortest_distance = {}
        track_predecessor = {}
        unseenNodes = graph
        infinity = np.Inf

        track_path = []

        # print(unseenNodes)
        for node in unseenNodes:
            # print(node)
            shortest_distance[node] = infinity
        shortest_distance[start] = 0

        while unseenNodes:
            min_dist_node = None

            for node in unseenNodes:
                if min_dist_node is None:
                    min_dist_node = node
                elif shortest_distance[node] < shortest_distance[min_dist_node]:
                    min_dist_node = node

            path_options = graph[min_dist_node].items()

            for child_node, weight in path_options:
                if weight + shortest_distance[min_dist_node] < shortest_distance[child_node]:
                    shortest_distance[child_node] = weight + shortest_distance[min_dist_node]
                    track_predecessor[child_node] = min_dist_node

            unseenNodes.pop(min_dist_node)

        currentNode = goal

        while currentNode != start:
            track_path.insert(0, currentNode)
            currentNode = track_predecessor[currentNode]

        track_path.insert(0, start)

        if shortest_distance[goal] != infinity:
            #print("--------------------------------------------------------------------------------------")
            #print("Shortest distnace is : " + str(shortest_distance[goal]))
            #print("Optimal path is " + str(track_path))

            return track_path
        else:
            print("Target not reachable.")
