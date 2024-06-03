class Graph:
    def __init__(self, num_verts, vertices=None):
        '''
        :param vertices: the number of vertices
        :param nodes:
        '''
        # Number of vertices
        self.num_verts = num_verts
        if vertices is not None:
            assert isinstance(vertices, list), 'vertices should be a list'
            self.vertices = vertices.copy()
        else:
            self.vertices = [i for i in range(self.num_verts)]

        # default dictionary to store graph
        self.graph = {index: set() for index in range(self.num_verts)}

    # function to add an edge to graph
    def add_edge(self, u, v, is_index=True, dual=True):
        '''
        :param u: the end of the edge
        :param v: the other end of the edge
        :param is_index: True  -> u,v is the index of the vertice /
                         False -> u,v is the value of the vertice
        :param dual: True -> undirected graph / False -> directed graph
        :return:
        '''
        assert u != v, 'u !=v in Graph'

        if is_index:
            uindex, vindex = u, v
        else:
            assert u in self.vertices, "{} is not in vertices".format(u)
            assert v in self.vertices, "{} is not in vertices".format(v)
            uindex, vindex = self.vertices.index(u), self.vertices.index(v)

        self.graph[uindex].add(vindex)
        if dual:
            self.graph[vindex].add(uindex)

    def del_edge(self, u, v, is_index=True, dual=True):
        assert u != v, 'u !=v in Graph'
        if is_index:
            uindex, vindex = u, v
        else:
            assert u in self.vertices, "{} is not in vertices".format(u)
            assert v in self.vertices, "{} is not in vertices".format(v)
            uindex, vindex = self.vertices.index(u), self.vertices.index(v)

        self.graph[uindex].remove(vindex)
        if dual:
            self.graph[vindex].remove(uindex)

    ## function to get all edges
    def get_edges(self, is_index=True):
        '''
        :param is_index: True   -> get the index of two points in an edge
                         False  -> get the value of two points in an edge
        :return:
        '''
        edges = set()
        for uindex in self.graph:
            for vindex in self.graph[uindex]:
                edges.add((uindex, vindex))

        edges = list(edges)
        result = []
        if not is_index:
            for edge in edges:
                result.append([self.vertices[edge[0]], self.vertices[edge[1]]])
        else:
            result = edges

        return result

    def get_verts(self):
        return self.vertices

    def get_num_verts(self):
        return self.num_verts

    ## Deep first search
    def dfs(self, start, is_index=True):
        '''
        :return: the path of deepest path
        '''

        if not is_index:
            assert start in self.vertices, "{} is not in vertices".format(start)
            start = self.vertices.index(start)

        ## the vertices have been visited
        visited = list()
        longest = (0, [])

        def dfs_traverse(v):
            nonlocal longest
            ## this vertice has been visited
            visited.append(v)

            ## connection vertices
            verts = list(self.graph[v])
            verts = [v for v in verts if v not in visited]
            # print("visited: {} verts: {}".format(visited,verts))

            ## Cannot go on deeper search
            if len(verts) == 0:
                seq = visited.copy()
                if len(seq) > longest[0]:
                    longest = (len(seq), seq)
                visited.pop()
                return

            ## Traverse All connected vertices
            for vert in verts:
                dfs_traverse(vert)
            visited.pop()

        dfs_traverse(start)
        return longest

    def get_acyclic_graph(self):
        '''
        dfs, if v has been visited, there is a circle
        :return:
        '''

        acyclic_graph = Graph(self.num_verts, self.vertices)

        ## the vertices have been visited
        visited = list()
        abandon = list()

        def dfs_traverse(v, last):
            nonlocal abandon
            ## this vertice has been visited
            visited.append(v)

            ## connection vertices
            verts = list(self.graph[v])
            abandon += [[v, vert] for vert in verts if vert in visited and vert != last]
            verts = [vert for vert in verts if vert not in visited]
            # print("v: {} verts: {} visited:{}".format(v,verts,visited))

            ## Cannot go on deeper search
            if len(verts) == 0:
                visited.pop()
                return

            ## Traverse All connected vertices
            for vert in verts:
                ## all connected and not visited
                acyclic_graph.add_edge(u=v, v=vert)
                if [vert, v] in abandon or [v, vert] in abandon:
                    continue
                dfs_traverse(vert, v)
            visited.pop()

        dfs_traverse(0, None)

        for edge in abandon:
            acyclic_graph.del_edge(u=edge[0], v=edge[1])

        return acyclic_graph

    def get_prunned_graph(self, threshold):
        ins_points = self.get_key_points(mode='ins')
        end_points = self.get_key_points(mode='end')
        key_points = ins_points + end_points

        ## immediately return once hit the key point
        def dfs_hit(start):
            ## connected points from end to end/ins
            connect_points = list()

            ## the vertices have been visited
            visited = list()

            def dfs_traverse(v):
                connect_points.append(v)
                if v in key_points and v != start:
                    return

                ## this vertice has been visited
                visited.append(v)

                ## connection vertices
                verts = list(self.graph[v])
                verts = [v for v in verts if v not in visited]

                ## Cannot go on deeper search
                if len(verts) == 0:
                    visited.pop()
                    return

                ## Traverse All connected vertices
                for vert in verts:
                    dfs_traverse(vert)
                visited.pop()

            dfs_traverse(start)
            return connect_points

        ## traverse all end point
        abandon_points = []
        for pnt in end_points:
            connect_points = dfs_hit(pnt)
            ## prun short section
            if len(connect_points) < threshold:
                abandon_points += list(set(connect_points) - set(ins_points))

        reserved_points = list(set([i for i in range(self.num_verts)]) - set(abandon_points))
        values = [self.vertices[i] for i in reserved_points]

        prunned_graph = Graph(num_verts=len(reserved_points), vertices=values)

        edges = self.get_edges()
        for edge in edges:
            u, v = edge[0], edge[1]
            if u in abandon_points or v in abandon_points:
                continue
            prunned_graph.add_edge(u=self.vertices[u], v=self.vertices[v], is_index=False)

        return prunned_graph

    ## get key points of the current graph (skeleton)
    def get_key_points(self, mode='all', is_index=True):
        '''
        :param dis_func:   calculate the distance between two node of the graph   (default: None)
        :param threshold:  if < distance, the point will be judge as the same one (default: None)
        :param mode: 'all' -> (intersection point + end point) [default]
                     'end' -> (end point)
                     'ins' -> (intersection point)
        :return:
        '''
        result = []
        for v in self.graph:
            if (mode == 'all' or mode == 'ins') and len(self.graph[v]) > 2:
                result.append(v)

            if (mode == 'all' or mode == 'end') and len(self.graph[v]) == 1:
                result.append(v)

        if not is_index:
            result = [self.vertices[i] for i in result]

        return result

    ## get key graph of the current graph (skeleton)
    def get_key_graph(self):
        key_points = self.get_key_points(mode='all')

        ## immediately return once hit the key point
        def dfs_hit(start):
            ## connected points
            connect_points = set()

            ## the vertices have been visited
            visited = list()

            def dfs_traverse(v):
                if v in key_points and v != start:
                    connect_points.add(v)
                    return

                ## this vertice has been visited
                visited.append(v)

                ## connection vertices
                verts = list(self.graph[v])
                verts = [v for v in verts if v not in visited]

                ## Cannot go on deeper search
                if len(verts) == 0:
                    visited.pop()
                    return

                ## Traverse All connected vertices
                for vert in verts:
                    dfs_traverse(vert)
                visited.pop()

            dfs_traverse(start)
            return list(connect_points)

        key_graph = Graph(len(key_points), [self.vertices[i] for i in key_points])
        for uindex, pnt in enumerate(key_points):
            connected_points = dfs_hit(pnt)
            for cp in connected_points:
                vindex = key_points.index(cp)
                key_graph.add_edge(u=uindex, v=vindex)

            # for cp in connected_points:
            #     vindex = key_points.index(cp)
            #     key_graph.add_edge(u=uindex, v=vindex)

        return key_graph

    def save_graph(self, points_path, edges_path):
        np.save(points_path, self.vertices)
        np.save(edges_path, self.get_edges())

    @staticmethod
    def load_graph(points_path, edges_path):
        vertices = np.load(points_path)
        graph = Graph(len(vertices), vertices.tolist())
        edges = np.load(edges_path)
        for edge in edges:
            graph.add_edge(u=edge[0], v=edge[1])

        return graph

    ## show graph
    def __repr__(self):
        str = ""
        for node in self.graph:
            str += "[{}]->{}\n".format(node, self.graph[node])
        return str


if __name__ == '__main__':
    import numpy as np
    from scipy.ndimage import binary_fill_holes, binary_closing
    from mtool.mutils.mbinary import get_largest_n_connected_region
    import matplotlib.pyplot as plt

    mipo = np.load('../../mip-dai.npy')

    hist, bins = np.histogram(mipo.ravel())
    threshold = bins[-3]
    binmip = mipo > threshold
    binmip = binary_closing(binmip)
    binmip = binary_fill_holes(binmip)
    binmip = get_largest_n_connected_region(binmip, 1)[0]

    plt.imshow(binmip, cmap=plt.cm.bone)

    ## Skeletonize
    from skimage.morphology import skeletonize
    skeleton = skeletonize(binmip)

    ## Generate the graph of the skeleton
    sk_point = np.argwhere(skeleton == 1).tolist()
    plt.scatter(np.asarray(sk_point)[:, 1], np.asarray(sk_point)[:, 0], s=0.1, color='pink')
    sk_graph = Graph(num_verts=len(sk_point), vertices=sk_point)
    for uindex, pnt in enumerate(sk_point):
        ## find 4/8 neighbor
        four_neighbor = [
            [pnt[0] - 1, pnt[1]],
            [pnt[0] + 1, pnt[1]],
            [pnt[0], pnt[1] - 1],
            [pnt[0], pnt[1] + 1]
        ]

        eight_neighbor = four_neighbor + [
            [pnt[0] - 1, pnt[1] - 1],
            [pnt[0] + 1, pnt[1] + 1],
            [pnt[0] + 1, pnt[1] - 1],
            [pnt[0] - 1, pnt[1] + 1]
        ]

        neighbors = eight_neighbor

        for neighor in neighbors:
            if neighor in sk_point:
                vindex = sk_point.index(neighor)
                sk_graph.add_edge(uindex, vindex)
                # print("pnt:{} neighbor:{} index:{} index neighbor:{}".format(pnt,neighor,index,sk_point[index]))


    ## polymerization - calculate distance function
    def dis_func(src_pnt, dst_pnt):
        w = (src_pnt[0] - dst_pnt[0]) ** 2
        h = (src_pnt[1] - dst_pnt[1]) ** 2
        return np.sqrt(w + h)


    ## polymeric graph
    sk_graph = sk_graph.get_acyclic_graph()
    sk_graph.save_graph(points_path='./point.npy', edges_path='./edge.npy')

    import numpy as np
    import matplotlib.pyplot as plt

    sk_graph = Graph.load_graph(points_path='./point.npy', edges_path='./edge.npy')

    sk_graph = sk_graph.get_acyclic_graph()
    sk_graph = sk_graph.get_prunned_graph(threshold=20)
    sk_graph = sk_graph.get_key_graph()
    # sk_graph = sk_graph.get_prunned_graph(threshold=50)
    # sk_graph = sk_graph.get_prunned_graph(threshold=70)

    end_point = sk_graph.get_key_points(mode='end', is_index=False)
    # print(end_point)
    # print(len(end_point))
    # end_point = sk_graph.vertices
    end_point = np.asarray(end_point)
    plt.scatter(end_point[:, 1], end_point[:, 0], s=10, color='blue')

    edges = sk_graph.get_edges(is_index=False)
    for edge in edges:
        left_point = edge[0]
        rigt_point = edge[1]

        x_point = [left_point[1], rigt_point[1]]
        y_point = [left_point[0], rigt_point[0]]
        plt.plot(x_point, y_point, linewidth=1, color='red')
    #
    # sk_point = np.asarray([sk_point[i] for i in key_points])
    # plt.scatter(sk_point[:, 1], sk_point[:, 0], s=3, color='green')
    plt.axis('off')
    plt.savefig('./test.png', dpi=200)
    plt.show()
