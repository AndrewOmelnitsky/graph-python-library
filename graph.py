from abc import ABC, abstractmethod
import math
from math import cos, sin, atan, sqrt
import pygame
import copy
import numpy as np
from pygame.locals import *
from numba import njit, jit
pygame.init()

class Graph(ABC):
    pass

class DirectedGraph(Graph):

        def __init__(self, start_nodes: list[int], end_nodes: list[int],
            edges_weight_list: list[float] = None,
            nodes_weight_list: list[float] = None) -> None:

            self.convert_se_list_to_edges_list(start_nodes, end_nodes)
            self.convert_se_list_to_connection_list(start_nodes, end_nodes)
            if edges_weight_list is not None:
                if len(start_nodes) != len(edges_weight_list):
                    raise Exception('wrong length of edges weight list')
                self.edges_weight_list = [-1 for i in range(len(edges_weight_list))]
                for i in range(len(edges_weight_list)):
                    self.update_weight_for_edge(i, edges_weight_list[i])

            if nodes_weight_list is not None:
                if len(start_nodes) != len(nodes_weight_list):
                    raise Exception('wrong length of nodes weight list')
                self.nodes_weight_list = [-1 for i in range(len(nodes_weight_list))]
                for i in range(len(nodes_weight_list)):
                    self.update_weight_for_node(i, nodes_weight_list[i])


        def get_childre_by_node(self, node):
            if node in self.connection_list:
                return self.connection_list[node]
            else:
                raise Exception('wrong node number')

                
        def update_weight_for_edge(self, edge: int, weight: float):
            if not len(self.edges_weight_list) > edge >= 0:
                raise Exception(f"fail to update weight. edge not in edges list. wrong edge number: {edge}")
            self.edges_weight_list[edge] = weight

        def update_weight_for_node(self, node: int, weight: float):
            if not len(self.nodes_weight_list) > node >= 0:
                raise Exception(f"fail to update weight. edge not in edges list. wrong edge number: {edge}")
            self.nodes_weight_list[node] = weight

        def convert_se_list_to_edges_list(self, start_nodes: list[int], end_nodes: list[int]):
            self.edges_list = list()
            if len(start_nodes) != len(end_nodes):
                raise Exception('wrong length of start or end nodes list')
            for i in range(len(start_nodes)):
                self.edges_list.append((start_nodes[i], end_nodes[i]))

        def convert_se_list_to_connection_list(self, start_nodes: list[int], end_nodes: list[int]):
            if len(start_nodes) != len(end_nodes):
                raise Exception('wrong length of start or end nodes list')
            self.connection_list = dict()
            for i in range(len(start_nodes)):
                if start_nodes[i] not in self.connection_list:
                    self.connection_list[start_nodes[i]] = list()
                if end_nodes[i] not in self.connection_list:
                    self.connection_list[end_nodes[i]] = list()

                self.connection_list[start_nodes[i]].append(end_nodes[i])

        def set_matrix(self, matrix: list[list[int]]) -> None:
            self.connection_matrix = matrix

        def set_connection_list(self, con_list: dict) -> None:
            self.connection_list = con_list



class UndirectedGraph(Graph):

    def __init__(self, start_nodes: list[int], end_nodes: list[int],
        edges_weight_list: list[float] = None,
        nodes_weight_list: list[float] = None) -> None:

        self.convert_se_list_to_edges_list(start_nodes, end_nodes)
        self.convert_se_list_to_connection_list(start_nodes, end_nodes)
        self.edges_weight_list = None
        if edges_weight_list is not None:
            if len(start_nodes) != len(edges_weight_list):
                raise Exception('wrong length of edges weight list')
            self.edges_weight_list = [-1 for i in range(len(edges_weight_list))]
            for i in range(len(edges_weight_list)):
                self.update_weight_for_edge(i, edges_weight_list[i])

        self.nodes_weight_list = None
        if nodes_weight_list is not None:
            if len(self.get_nodes()) != len(nodes_weight_list):
                raise Exception('wrong length of nodes weight list')
            self.nodes_weight_list = [-1 for i in range(len(nodes_weight_list))]
            for i in range(len(nodes_weight_list)):
                self.update_weight_for_node(i, nodes_weight_list[i])


    def get_nodes(self):
        return list(self.connection_list.keys())

    def get_edges(self):
        return list(self.edges_list)

    def get_edge_id(self, edge):
        if self.dose_edge_in_graph(edge):
            if edge in self.edges_list:
                return self.edges_list.index(edge)
            return self.edges_list.index(edge[::-1])

        else:
            raise Exception(f'{edge=} not in graph')

    def get_incedent_edges(self, node):
        res_edges = []
        for edge in self.get_edges():
            if node in edge:
                res_edges.append(edge)

        return res_edges

    def dose_node_in_graph(self, node):
        return node in self.connection_list

    def dose_edge_in_graph(self, edge):
        dose_include = edge in self.edges_list
        dose_include_reverse = edge[::-1] in self.edges_list
        return any((dose_include, dose_include_reverse))

    def get_childre_by_node(self, node):
        if node in self.connection_list:
            return self.connection_list[node]
        else:
            raise Exception(f'wrong node number: {node}')

    def get_weight_by_edge(self, edge):
        if self.edges_weight_list is None:
            raise Exception('this graph is unweighted')

        if edge in self.edges_list:
            return self.edges_weight_list[self.edges_list.index(edge)]

        reverse_edge = edge[::-1]
        if reverse_edge in self.edges_list:
            return self.edges_weight_list[self.edges_list.index(reverse_edge)]

        raise Exception(f'wrong edge: {edge}')
        return None

    def get_weight_by_node(self, node):
        if self.nodes_weight_list is None:
            raise Exception('this graph is unweighted')

        if node in self.get_nodes():
            return self.nodes_weight_list[node-1]

        raise Exception(f'wrong node: {node}')
        return None



    def update_weight_for_edge(self, edge: int, weight: float):
        if not len(self.edges_weight_list) > edge >= 0:
            raise Exception(f"fail to update weight. edge not in edges list. wrong edge number: {edge}")
        self.edges_weight_list[edge] = weight

    def update_weight_for_node(self, node: int, weight: float):
        if not len(self.nodes_weight_list) > node >= 0:
            raise Exception(f"fail to update weight. edge not in edges list. wrong edge number: {edge}")
        self.nodes_weight_list[node] = weight

    def convert_se_list_to_edges_list(self, start_nodes: list[int], end_nodes: list[int]):
        self.edges_list = list()
        if len(start_nodes) != len(end_nodes):
            raise Exception('wrong length of start or end nodes list')
        for i in range(len(start_nodes)):
            self.edges_list.append((start_nodes[i], end_nodes[i]))

    def convert_se_list_to_connection_list(self, start_nodes: list[int], end_nodes: list[int]):
        if len(start_nodes) != len(end_nodes):
            raise Exception('wrong length of start or end nodes list')
        self.connection_list = dict()
        for i in range(len(start_nodes)):
            if start_nodes[i] not in self.connection_list:
                self.connection_list[start_nodes[i]] = list()
            if end_nodes[i] not in self.connection_list:
                self.connection_list[end_nodes[i]] = list()

            self.connection_list[start_nodes[i]].append(end_nodes[i])
            self.connection_list[end_nodes[i]].append(start_nodes[i])

    def set_matrix(self, matrix: list[list[int]]) -> None:
        self.connection_matrix = matrix

    def set_connection_list(self, con_list: dict) -> None:
        self.connection_list = con_list


def draw_arrow_old(surface: pygame.Surface,
                   color: set[int],
                   point1: set[int],
                   point2: set[int],
                   pointer_len: int = 5,
                   pointer_position: int = 0):
    p = np.array((point2[0] - point1[0], point2[1] - point1[1]))
    p_len = sqrt(p[0]**2 + p[1]**2)

    theta1 = np.deg2rad(45)
    rot1 = np.array([[cos(theta1), -sin(theta1)], [sin(theta1), cos(theta1)]])
    theta2 = np.deg2rad(-45)
    rot2 = np.array([[cos(theta2), -sin(theta2)], [sin(theta2), cos(theta2)]])

    #print(-np.dot(rot1, p*pointer_len/p_len) + np.array(point2))
    pointer_point =  np.array(point2)
    if pointer_position:
        pointer_point = (np.array(point2) + np.array(point1)) / 2
    a1 = -np.dot(rot1, p * pointer_len / p_len) + pointer_point
    a2 = -np.dot(rot2, p * pointer_len / p_len) + pointer_point

    pygame.draw.line(surface, color, point1, point2, 2)
    pygame.draw.line(surface, color, a1, pointer_point, 2)
    pygame.draw.line(surface, color, a2, pointer_point, 2)


def draw_arrow(surface: pygame.Surface,
                color: set[int],
                point1: set[int],
                point2: set[int],
                pointer_len: int = 6,
                alpha: int = 45,
                border: int = 2,
                pointer_position: int = 0):
    p = np.array((point2[0] - point1[0], point2[1] - point1[1]))
    p_len = sqrt(p[0]**2 + p[1]**2)

    theta1 = np.deg2rad(alpha)
    rot1 = np.array([[cos(theta1), -sin(theta1)], [sin(theta1), cos(theta1)]])
    theta2 = np.deg2rad(-alpha)
    rot2 = np.array([[cos(theta2), -sin(theta2)], [sin(theta2), cos(theta2)]])

    pointer_point = np.array(point2)
    if pointer_position:
        #print('-'*20)
        pointer_point = (np.array(point2) + np.array(point1)) / 2

    a1 = -np.dot(rot1, p * pointer_len / p_len) + pointer_point
    a2 = -np.dot(rot2, p * pointer_len / p_len) + pointer_point
    #print(point2)
    #print(pointer_point)

    pygame.draw.line(surface, color, point1, point2, border)
    pygame.draw.polygon(surface, color, (a1, a2, pointer_point))


class UndirectedGraphAlgorithms(object):
    def __init__(self, graph: Graph):
        self.graph = graph

    def max_match(self):
        graph = self.graph
        if graph.edges_weight_list is not None:
            def count_all_weight(graph, edges):
                all_weight = 0
                for edge in edges:
                    all_weight += graph.get_weight_by_edge(edge)

                return all_weight


            def helper(graph, node, next_verts, edges):
                result_edges = list()
                if len(next_verts) < 2:
                    return edges
                if node in next_verts:
                    next_verts.pop(next_verts.index(node))
                for del_node in graph.get_childre_by_node(node):
                    if del_node not in next_verts:
                        continue
                    new_next_verts = next_verts[:]
                    new_next_verts.pop(new_next_verts.index(del_node))
                    new_edges = edges[:]
                    new_edges.append((node, del_node))
                    if len(new_edges) > len(result_edges):
                        result_edges = new_edges
                    elif len(new_edges) == len(result_edges):
                        if count_all_weight(graph, new_edges) > count_all_weight(graph, result_edges):
                            result_edges = new_edges
                    for next_node in new_next_verts:

                        temp_edges = helper(graph, next_node, new_next_verts, new_edges)
                        #print(temp_edges)
                        if len(temp_edges) > len(result_edges):
                            result_edges = temp_edges
                        elif len(temp_edges) == len(result_edges):
                            if count_all_weight(graph, temp_edges) > count_all_weight(graph, result_edges):
                                result_edges = temp_edges
                return result_edges


            result_edges = list()
            for start_node in graph.get_nodes():
                next_verts = graph.get_nodes()
                next_verts.pop(next_verts.index(start_node))
                temp_edges = helper(graph, start_node, next_verts, [])
                if len(temp_edges) > len(result_edges):
                    result_edges = temp_edges
                elif len(temp_edges) == len(result_edges):
                    if count_all_weight(graph, temp_edges) > count_all_weight(graph, result_edges):
                        result_edges = temp_edges

            # s = [item[0] for item in result_edges]
            # e = [item[1] for item in result_edges]
            # result_graph = UndirectedGraph(s, e)
            return result_edges, count_all_weight(graph, result_edges)
        else:
            def helper(graph, node, next_verts, edges):
                result_edges = list()
                if len(next_verts) < 2:
                    return edges
                if node in next_verts:
                    next_verts.pop(next_verts.index(node))
                for del_node in graph.get_childre_by_node(node):
                    if del_node not in next_verts:
                        continue
                    new_next_verts = next_verts[:]
                    new_next_verts.pop(new_next_verts.index(del_node))
                    new_edges = edges[:]
                    new_edges.append((node, del_node))
                    if len(new_edges) > len(result_edges):
                        result_edges = new_edges
                    for next_node in new_next_verts:

                        temp_edges = helper(graph, next_node, new_next_verts, new_edges)
                        #print(temp_edges)
                        if len(temp_edges) > len(result_edges):
                            result_edges = temp_edges
                return result_edges


            result_edges = list()
            for start_node in graph.get_nodes():
                next_verts = graph.get_nodes()
                next_verts.pop(next_verts.index(start_node))
                temp_edges = helper(graph, start_node, next_verts, [])
                if len(temp_edges) > len(result_edges):
                    result_edges = temp_edges

            # s = [item[0] for item in result_edges]
            # e = [item[1] for item in result_edges]
            # result_graph = UndirectedGraph(s, e)
            return result_edges, len(result_edges)

    def max_indset(self):
        graph = self.graph
        if graph.nodes_weight_list is not None:
            def count_all_weight(graph, nodes):
                all_weight = 0
                for node in nodes:
                    all_weight += graph.get_weight_by_node(node)

                return all_weight

            def helper(graph, node, next_verts, nodes):
                # print('-'*10)
                # print(f'{node=}')
                # print(f'{next_verts=}')
                # print(f'{nodes=}')
                # print('-'*10)
                result_nodes = list()
                nodes.append(node)

                if node in next_verts:
                    next_verts.pop(next_verts.index(node))

                if len(next_verts) == 0:
                    return nodes



                for del_node in graph.get_childre_by_node(node):
                    if del_node not in next_verts:
                        continue
                    next_verts.pop(next_verts.index(del_node))

                if len(nodes) > len(result_nodes):
                    result_nodes = nodes
                elif len(nodes) == len(result_nodes):
                    if count_all_weight(graph, nodes) > count_all_weight(graph, result_nodes):
                        result_nodes = nodes

                for next_node in next_verts:
                    temp_nodes = helper(graph, next_node, next_verts, nodes)
                    if len(temp_nodes) > len(result_nodes):
                        result_nodes = temp_nodes
                    elif len(temp_nodes) == len(result_nodes):
                        if count_all_weight(graph, temp_nodes) > count_all_weight(graph, result_nodes):
                            result_nodes = temp_nodes
                return result_nodes


            result_nodes = list()
            for start_node in sorted(graph.get_nodes()):
                next_verts = graph.get_nodes()
                temp_nodes = helper(graph, start_node, next_verts, [])
                if len(temp_nodes) > len(result_nodes):
                    result_nodes = temp_nodes
                elif len(temp_nodes) == len(result_nodes):
                    if count_all_weight(graph, temp_nodes) > count_all_weight(graph, result_nodes):
                        result_nodes = temp_nodes

            return result_nodes, count_all_weight(graph, result_nodes)
        else:
            def helper(graph, node, next_verts, nodes):
                # print('-'*10)
                # print(f'{node=}')
                # print(f'{next_verts=}')
                # print(f'{nodes=}')
                # print('-'*10)
                result_nodes = list()
                nodes.append(node)

                if node in next_verts:
                    next_verts.pop(next_verts.index(node))

                if len(next_verts) == 0:
                    return nodes



                for del_node in graph.get_childre_by_node(node):
                    if del_node not in next_verts:
                        continue
                    next_verts.pop(next_verts.index(del_node))

                if len(nodes) > len(result_nodes):
                    result_nodes = nodes

                for next_node in next_verts:
                    temp_nodes = helper(graph, next_node, next_verts, nodes)
                    if len(temp_nodes) > len(result_nodes):
                        result_nodes = temp_nodes
                return result_nodes


            result_nodes = list()
            for start_node in sorted(graph.get_nodes()):
                next_verts = graph.get_nodes()
                temp_edges = helper(graph, start_node, next_verts, [])
                if len(temp_edges) > len(result_nodes):
                    result_nodes = temp_edges

            return result_nodes, len(result_nodes)


    def min_edge_cover(self):
        graph = self.graph
        edges, _ = self.max_match()
        visited_nodes = set()
        for edge in edges:
            visited_nodes.add(edge[0])
            visited_nodes.add(edge[1])

        all_nodes = set(graph.get_nodes())
        unvisited_nodes = all_nodes - visited_nodes

        if graph.edges_weight_list is not None:
            for node in unvisited_nodes:
                ch = graph.get_childre_by_node(node)
                if len(ch):
                    min_weight = math.inf
                    min_edge = None
                    for end_node in graph.get_childre_by_node(node):
                        edge = (node, end_node)
                        if min_weight > graph.get_weight_by_edge(edge):
                            min_edge = edge
                            min_weight = graph.get_weight_by_edge(edge)
                    edges.append(min_edge)
        else:
            for node in unvisited_nodes:
                ch = graph.get_childre_by_node(node)
                if len(ch):
                    edges.append((node, ch[0]))

        return edges, len(edges)


    def min_vert_cover(self):
        graph = self.graph

        def dose_swap(nodes1, nodes2):
            if len(nodes1) < len(nodes2):
                return True
            return False

        def get_second_param(nodes):
            return len(nodes)

        if graph.nodes_weight_list is not None:
            def count_all_weight(graph, nodes):
                all_weight = 0
                for node in nodes:
                    all_weight += graph.get_weight_by_node(node)

                return all_weight
            def new_dose_swap(nodes1, nodes2):
                if len(nodes1) < len(nodes2):
                    return True
                elif len(nodes1) == len(nodes2):
                    if count_all_weight(graph, nodes1) < count_all_weight(graph, nodes2):
                        return True

                return False

            def new_get_second_param(nodes):
                print(f'{nodes=}')
                print(f'nodes all weight={count_all_weight(graph, nodes)}')
                return count_all_weight(graph, nodes)

            get_second_param = new_get_second_param
            dose_swap = new_dose_swap

        def helper(graph, node, next_nodes, nodes = [], visited_edges = set(), nodes_lists = dict()):
            result_nodes = graph.get_nodes()
            if len(visited_edges) == len(graph.get_edges()):
                return nodes

            all_edges_in_node = set(graph.get_incedent_edges(node))
            if len(visited_edges & all_edges_in_node) == len(all_edges_in_node):
                return result_nodes

            if node in next_nodes:
                next_nodes.pop(next_nodes.index(node))

            nodes.append(node)

            for edge in all_edges_in_node:
                visited_edges.add(edge)

            for next_node in next_nodes:
                temp_nodes = helper(graph, next_node, next_nodes[:], nodes[:], visited_edges.copy(), nodes_lists)
                if dose_swap(temp_nodes, result_nodes):
                    result_nodes = temp_nodes

            return result_nodes

        result_nodes = graph.get_nodes()
        nodes_lists = list()
        for start_node in graph.get_nodes():
            temp_nodes = helper(graph, start_node, graph.get_nodes(), [], set(), nodes_lists)
            print(f'{start_node=}')
            print(f'{temp_nodes=}')
            if dose_swap(temp_nodes, result_nodes):
                result_nodes = temp_nodes

        return result_nodes, get_second_param(result_nodes)

    def min_dom_edge_set(self):
        graph = self.graph

        @njit
        def dose_swap(edges1, edges2):
            l1 = sum(edges1)
            l2 = sum(edges2)

            if l1 < l2:
                return True

        def get_second_param(nodes):
            return len(nodes)

        if graph.edges_weight_list is not None:
            weights = np.array(graph.edges_weight_list)

            @njit
            def count_all_weight(edges):
                all_weight = 0
                for i in range(len(edges)):
                    if edges[i]:
                        all_weight += weights[i]

                return all_weight


            @njit
            def new_dose_swap(edges1, edges2):
                l1 = sum(edges1)
                l2 = sum(edges2)

                if l1 < l2:
                    return True
                elif l1 == l2:
                    if count_all_weight(edges1) < count_all_weight(edges2):
                        return True

                return False

            def new_get_second_param(edges):
                print(f'{edges=}')
                print(f'edges all weight={count_all_weight(edges)}')
                return count_all_weight(edges)

            get_second_param = new_get_second_param
            dose_swap = new_dose_swap

        @jit
        def helper(all_edges, edge, next_edges, edges, visited_nodes):
            result_edges: np.array = np.ones(len(edges), dtype='int64')

            if 0 not in visited_nodes:
                return edges

            if visited_nodes[edge[0]-1] and visited_nodes[edge[1]-1]:
                return result_edges

            edge_id: int = 0
            for i in range(len(all_edges)):
                if edge[0] == all_edges[i][0] and edge[1] == all_edges[i][1]:
                    edge_id = i
                    break

            next_edges[edge_id] = 0
            edges[edge_id] = 1

            visited_nodes[edge[0]-1] = 1
            visited_nodes[edge[1]-1] = 1

            for i in range(len(next_edges)):
                if next_edges[i]:
                    new_all_edges = all_edges
                    next_edge = all_edges[i]
                    new_next_edges = np.copy(next_edges)
                    new_edges = np.copy(edges)
                    new_visited_nodes = np.copy(visited_nodes)
                    temp_edges: np.array = helper(all_edges, next_edge, new_next_edges, new_edges, new_visited_nodes)
                    if dose_swap(temp_edges, result_edges):
                        result_edges = temp_edges

            return result_edges

        result_edges = graph.get_edges()
        all_edges = np.array(graph.get_edges())
        n_edges = np.ones(len(graph.get_edges()), dtype='bool')
        edges = np.zeros(len(graph.get_edges()), dtype='bool')
        visited_nodes = np.zeros(len(graph.get_nodes()), dtype='bool')
        for start_edge in graph.get_edges():
            temp_edges = helper(np.copy(all_edges), np.array(start_edge), np.copy(n_edges), np.copy(edges), np.copy(visited_nodes))
            print(f'{start_edge=}')
            print(f'{temp_edges=}')
            if dose_swap(temp_edges, result_edges):
                result_edges = temp_edges

        return result_edges, get_second_param(result_edges)




class GraphDrawer(object):
    node_color = (100, 100, 200)
    line_color = (150, 150, 255)
    border_color = (0, 0, 0)
    line_border = 2
    node_radius = 15
    node_border = 5
    border_radius = 5
    marked_edge_width = 10
    marked_edge_color = (200, 100, 100)
    marked_node_radius = 20
    marked_node_color = (150, 100, 100)
    background_color = (200, 200, 200)
    window_padding = 30
    font = pygame.font.Font(None, 24)

    def __init__(self, graph: Graph, grid_size: tuple[int],
        window_size: tuple[int],
        drawing_map: tuple[tuple[int]] = None,
        nodes_positions: dict = None):

        self.graph = graph
        self.grid_size = [i - 1 for i in grid_size]
        self.window_size = window_size
        self.surface = pygame.Surface(window_size)
        self.marked_nodes = dict()
        self.marked_edges = dict()
        if nodes_positions is not None:
            self.nodes_positions = nodes_positions
        else:
            if drawing_map is not None:
                self.update_nodes_positions_by_matrix(drawing_map)
            else:
                raise Exception('no drawing maps or node positions')
        #print(type(graph))
        if isinstance(graph, DirectedGraph):
            self.draw = self.draw_directed_graph
        elif isinstance(graph, UndirectedGraph):
            self.draw = self.draw_undirected_graph


    def resize(self, new_size: tuple[int]):
        if not isinstance(new_size, tuple):
            raise Exception(f'new size in not tuple. new size has type: {type(new_size)}')
        if len(new_size) != 2:
            raise Exception(f'length of new size tuple != 2 {new_size=}')

        self.surface = pygame.Surface(new_size)
        self.window_size = new_size

    def set_marked_nodes(self, marked_nodes):
        self.marked_nodes = dict()
        for node in marked_nodes:
            if self.graph.dose_node_in_graph(node):
                self.marked_nodes[node] = marked_nodes[node]

    def set_marked_edges(self, marked_edges):
        self.marked_edges = dict()
        for edge in marked_edges:
            if self.graph.dose_edge_in_graph(edge):
                self.marked_edges[edge] = marked_edges[edge]

    def update_nodes_positions_by_matrix(self, matrix):
        self.nodes_positions = dict()

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    self.nodes_positions[matrix[i][j]] = (j, i)

    def get_position_by_node(self, node):
        if node in self.nodes_positions:
            position = (
                self.nodes_positions[node][0]/self.grid_size[0]*(self.window_size[0] - 2*self.window_padding) + self.window_padding,
                self.nodes_positions[node][1]/self.grid_size[1]*(self.window_size[1] - 2*self.window_padding) + self.window_padding,
            )
            return position
        else:
            raise Exception('wrong node number')

    def draw_undirected_edge(self,
        start_position, end_position, lable, color, border, is_lable = True):

        if is_lable:
            text = self.font.render(str(lable), True, (0, 0, 0))
            place = text.get_rect(center=(np.array(start_position) + np.array(end_position))/2+10)

            p = (np.array(end_position) -  np.array(start_position))
            ox = np.array((1, 0))
            scalar = np.dot(p, ox)
            mod1 = np.sqrt(p.dot(p))
            mod2 = np.sqrt(ox.dot(ox))
            angle = (-1 if p[1] > 0 else 1) * math.degrees(math.acos(scalar/(mod1*mod2)))

            text = pygame.transform.rotate(text, angle)
            self.surface.blit(text, place)

        pygame.draw.line(self.surface,
            color,
            start_position,
            end_position,
            border)


    def draw_undirected_graph(self):
        self.surface.fill(self.background_color)
        get_lable = self.graph.get_edge_id
        if self.graph.edges_weight_list is not None:
            get_lable = self.graph.get_weight_by_edge
        for edge in self.graph.get_edges():
            start_position = self.get_position_by_node(edge[0])
            end_position = self.get_position_by_node(edge[1])
            self.draw_undirected_edge(start_position,
                end_position,
                get_lable(edge),
                self.line_color,
                self.line_border)

        for edge in self.marked_edges:
            start_position = self.get_position_by_node(edge[0])
            end_position = self.get_position_by_node(edge[1])
            self.draw_undirected_edge(start_position,
                end_position,
                get_lable(edge),
                self.marked_edges[edge],
                self.marked_edge_width)

        self.draw_nodes()
        self.draw_marked_nodes()
        self.draw_border()


    def draw_directed_graph(self):
        self.surface.fill(self.background_color)
        for start_node in self.nodes_positions:
            c_nodes = self.graph.get_childre_by_node(start_node)
            start_position = self.get_position_by_node(start_node)
            for end_node in c_nodes:
                end_position = self.get_position_by_node(end_node)
                draw_arrow(self.surface,
                    self.line_color,
                    start_position,
                    end_position,
                    pointer_position = 1)
            try:
                pass
            except Exception as e:
                print(e)
        self.draw_nodes()
        self.draw_border()


    def draw_border(self):
        pygame.draw.rect(self.surface, self.border_color, (0, 0, *self.window_size), 4, self.border_radius)


    def draw_node(self, node, lable, color, radius):
        node_position = self.get_position_by_node(node)
        pygame.draw.circle(self.surface,
            self.background_color,
            node_position,
            radius)
        pygame.draw.circle(self.surface,
            color,
            node_position,
            radius,
            self.node_border)
        text = self.font.render(str(lable), True, (0, 0, 0))
        place = text.get_rect(
            center=node_position)
        self.surface.blit(text, place)

    def draw_nodes(self):
        def get_node_id(node: int):
            return node
        get_lable = get_node_id
        if self.graph.nodes_weight_list is not None:
            get_lable = self.graph.get_weight_by_node
        for node in self.nodes_positions:
            self.draw_node(node, get_lable(node), self.node_color, self.node_radius)


    def draw_marked_nodes(self):
        def get_node_id(node: int):
            return node
        get_lable = get_node_id
        if self.graph.nodes_weight_list is not None:
            get_lable = self.graph.get_weight_by_node
        for node in self.marked_nodes:
            self.draw_node(node, get_lable(node), self.marked_nodes[node], self.marked_node_radius)

    def get_surface(self):
        return self.surface



def main():
    screen = pygame.display.set_mode((950, 500))
    is_run = True
    FPS = 60
    s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
    e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
    # w = [5, 5, 5, 2, 2, 3, 2, 5, 2, 3, 1, 1, 5, 2, 3, 2, 3, 2, 2, 5, 4, 5]
    w = [2, 3, 3, 4, 1, 2, 3, 3, 5, 1, 5]
    md = [
        [0, 1, 5, 8, 0],
        [2, 3, 6, 11, 9],
        [0, 4, 7, 10, 0],
    ]
    nd_pos = {
        1: (1, 0),
        5: (2, 0),
        8: (3, 0),
        2: (0, 1),
        3: (1, 1),
        6: (2, 1),
        11: (3, 1),
        9: (4, 1),
        4: (1, 2),
        7: (2, 2),
        10: (3, 2),
    }
    task_graph = UndirectedGraph(s, e, nodes_weight_list=w)
    gds = GraphDrawer(task_graph, (5, 3), (400, 400), nodes_positions=nd_pos)

    solve = UndirectedGraphAlgorithms(task_graph)

    result_nodes, max_weight = solve.max_indset()
    print(f'{max_weight=}')
    # result_edges, _, max_weight = solve.max_match()
    # marked_edges = {edge: gds.marked_edge_color for edge in result_edges}
    marked_nodes = {node: gds.marked_node_color for node in result_nodes}
    gds_res = GraphDrawer(task_graph, (5, 3), (400, 400), md)
    gds_res.set_marked_nodes(marked_nodes)

    while is_run:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                is_run = False

        screen.fill((255, 255, 255))

        #draw_arrow(screen, (0, 0, 0), (300, 100), (80, 200))#, alpha = 30, pointer_len = 10)
        gds_sur = gds.get_surface()
        gds_sur.fill((255, 255, 255))
        gds.draw()
        screen.blit(gds_sur, (50, 50))
        gds_sur_res = gds_res.get_surface()
        gds_sur_res.fill((255, 255, 255))
        gds_res.draw()
        screen.blit(gds_sur_res, (500, 50))

        pygame.display.update()
        pygame.time.delay(FPS)


if __name__ == "__main__":
    main()
