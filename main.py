from graph import *
import pygame

class TaskNotFound(Exception):
    pass

class SimpleGraphView(object):
    FPS = 60
    window_padding = 30

    def __init__(self, graph: Graph, graph_drawer: GraphDrawer,
        window_size: tuple[int]):

        graph_drawer.resize(tuple([item - self.window_padding*2 for item in window_size]))

        self.graph = graph
        self.graph_drawer = graph_drawer
        self.window_size = window_size


    def run(self):
        screen = pygame.display.set_mode(self.window_size)
        is_run = True
        FPS = 60
        while is_run:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    is_run = False

            screen.fill((255, 255, 255))
            self.graph_drawer.draw()
            screen.blit(self.graph_drawer.get_surface(),
                (self.window_padding, self.window_padding))

            pygame.display.update()
            pygame.time.delay(self.FPS)



class DualGraphView(object):
    FPS = 60
    window_padding = 30

    def __init__(self, graph: Graph,
        graph_drawer1: GraphDrawer,
        graph_drawer2: GraphDrawer,
        window_size: tuple[int]):

        graph_drawer_size = (
            (window_size[0] - self.window_padding*3)/2,
            window_size[1] - self.window_padding*2,)
        graph_drawer1.resize(graph_drawer_size)
        graph_drawer2.resize(graph_drawer_size)

        self.graph = graph
        self.graph_drawer_size = graph_drawer_size
        self.graph_drawer1 = graph_drawer1
        self.graph_drawer2 = graph_drawer2
        self.window_size = window_size


    def run(self):
        screen = pygame.display.set_mode(self.window_size)
        is_run = True
        FPS = 60
        while is_run:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    is_run = False

            screen.fill((255, 255, 255))

            self.graph_drawer1.draw()
            screen.blit(self.graph_drawer1.get_surface(),
                (self.window_padding, self.window_padding))

            self.graph_drawer2.draw()
            screen.blit(self.graph_drawer2.get_surface(),
                (self.window_padding*2 + self.graph_drawer_size[0], self.window_padding))

            pygame.display.update()
            pygame.time.delay(self.FPS)



class Homework(object):
    def __init__(self):
        self.tasks = {
            '1': self.task1,
            '1.1': self.task1_1,
            '2': self.task2,
            '2.1': self.task2_1,
            '3': self.task3,
            '3.1': self.task3_1,
            '4': self.task4,
            '4.1': self.task4_1,
            '5': self.task5,
            '5.1': self.task5_1,
        }


    def run_task(self, task: str):
        if not isinstance(task, str):
            raise ValueError()

        if task not in self.tasks:
            raise TaskNotFound(f'task with number {task} not found')

        self.tasks[task]()


    def task1(self):
        s = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 9, 10, 11]
        e = [2, 4, 5, 8, 3, 5, 5, 6, 8, 5, 7, 8, 6, 8, 8, 9, 8, 10, 9, 10, 11, 12, 12, 11, 12]
        # s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
        # e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
        w = [7, 5, 8, 4, 8, 3, 6, 2, 7, 5, 8, 7, 9, 6, 5, 4, 3, 3, 2, 4, 5, 6, 7, 8, 9]
        md = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]
        task_graph = UndirectedGraph(s, e, edges_weight_list=w)
        gds = GraphDrawer(task_graph, (3, 4), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_edges, max_weight = solve.max_match()
        print(f'{max_weight=}')
        marked_edges = {edge: gds.marked_edge_color for edge in result_edges}
        gds.set_marked_edges(marked_edges)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()

    def task1_1(self):
        s = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 9, 10, 11]
        e = [2, 4, 5, 8, 3, 5, 5, 6, 8, 5, 7, 8, 6, 8, 8, 9, 8, 10, 9, 10, 11, 12, 12, 11, 12]
        md = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]
        task_graph = UndirectedGraph(s, e)
        gds = GraphDrawer(task_graph, (3, 4), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_edges, max_len = solve.max_match()
        print(f'{max_len=}')
        marked_edges = {edge: gds.marked_edge_color for edge in result_edges}
        gds.set_marked_edges(marked_edges)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()


    def task2(self):
        s = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 9, 10, 11]
        e = [2, 4, 5, 8, 3, 5, 5, 6, 8, 5, 7, 8, 6, 8, 8, 9, 8, 10, 9, 10, 11, 12, 12, 11, 12]
        w = [10, 12, 8, 11, 6, 10, 16, 5, 13, 8, 10, 4]
        md = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]
        task_graph = UndirectedGraph(s, e, nodes_weight_list=w)
        gds = GraphDrawer(task_graph, (3, 4), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_nodes, max_weight = solve.max_indset()
        print(f'{max_weight=}')
        marked_nodes = {node: gds.marked_node_color for node in result_nodes}
        gds.set_marked_nodes(marked_nodes)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()

    def task2_1(self):
        s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
        e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
        md = [
            [0, 1, 5, 8, 0],
            [2, 3, 6, 11, 9],
            [0, 4, 7, 10, 0],
        ]
        task_graph = UndirectedGraph(s, e)
        gds = GraphDrawer(task_graph, (5, 3), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_nodes, max_len = solve.max_indset()
        print(f'{max_len=}')
        marked_nodes = {node: gds.marked_node_color for node in result_nodes}
        gds.set_marked_nodes(marked_nodes)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()

    def task3_1(self):
        s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
        e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
        md = [
            [0, 1, 5, 8, 0],
            [2, 3, 6, 11, 9],
            [0, 4, 7, 10, 0],
        ]
        task_graph = UndirectedGraph(s, e)
        gds = GraphDrawer(task_graph, (5, 3), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_edges, min_len = solve.min_edge_cover()
        print(f'{min_len=}')
        marked_edges = {edge: gds.marked_edge_color for edge in result_edges}
        gds.set_marked_edges(marked_edges)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()

    def task3(self):
        s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
        e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
        w = [5, 5, 5, 2, 2, 3, 2, 5, 2, 3, 1, 1, 5, 2, 3, 2, 3, 2, 2, 5, 4, 5]
        md = [
            [0, 1, 5, 8, 0],
            [2, 3, 6, 11, 9],
            [0, 4, 7, 10, 0],
        ]
        task_graph = UndirectedGraph(s, e, edges_weight_list=w)
        gds = GraphDrawer(task_graph, (5, 3), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_edges, min_weight = solve.min_edge_cover()
        print(f'{min_weight=}')
        marked_edges = {edge: gds.marked_edge_color for edge in result_edges}
        gds.set_marked_edges(marked_edges)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()

    def task4_1(self):
        s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
        e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
        md = [
            [0, 1, 5, 8, 0],
            [2, 3, 6, 11, 9],
            [0, 4, 7, 10, 0],
        ]
        task_graph = UndirectedGraph(s, e)
        gds = GraphDrawer(task_graph, (5, 3), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_nodes, min_len = solve.min_vert_cover()
        print(f'{min_len=}')
        marked_nodes = {node: gds.marked_node_color for node in result_nodes}
        gds.set_marked_nodes(marked_nodes)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()

    def task4(self):
        s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
        e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
        w = [2, 3, 3, 4, 1, 2, 3, 3, 5, 1, 5]
        md = [
            [0, 1, 5, 8, 0],
            [2, 3, 6, 11, 9],
            [0, 4, 7, 10, 0],
        ]
        task_graph = UndirectedGraph(s, e, nodes_weight_list=w)
        gds = GraphDrawer(task_graph, (5, 3), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_nodes, min_weight = solve.min_vert_cover()
        print(f'{min_weight=}')
        marked_nodes = {node: gds.marked_node_color for node in result_nodes}
        gds.set_marked_nodes(marked_nodes)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()

    def task5(self):
        s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
        e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
        w = [5, 5, 5, 2, 2, 3, 2, 5, 2, 3, 1, 1, 5, 2, 3, 2, 3, 2, 2, 5, 4, 5]
        md = [
            [0, 1, 5, 8, 0],
            [2, 3, 6, 11, 9],
            [0, 4, 7, 10, 0],
        ]
        task_graph = UndirectedGraph(s, e, edges_weight_list=w)
        gds = GraphDrawer(task_graph, (5, 3), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_edges, min_weight = solve.min_dom_edge_set()
        print(f'{min_weight=}')
        marked_edges = {edge: gds.marked_edge_color for edge in result_edges}
        gds.set_marked_edges(marked_edges)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()

    def task5_1(self):
        s = [2, 2, 2, 1, 3, 1, 1, 3, 3, 4, 5, 6, 5, 6, 6, 7, 7, 8, 11, 8, 11, 10]
        e = [1, 3, 4, 3, 4, 5, 6, 6, 7, 7, 6, 7, 8, 8, 11, 11, 10, 11, 10, 9, 9, 9]
        md = [
            [0, 1, 5, 8, 0],
            [2, 3, 6, 11, 9],
            [0, 4, 7, 10, 0],
        ]
        task_graph = UndirectedGraph(s, e)
        gds = GraphDrawer(task_graph, (5, 3), (400, 400), md)

        solve = UndirectedGraphAlgorithms(task_graph)

        result_edges, min_len = solve.min_dom_edge_set()
        print(f'{min_len=}')
        marked_edges = {edge: gds.marked_edge_color for edge in result_edges}
        gds.set_marked_edges(marked_edges)

        view = SimpleGraphView(task_graph, gds, (500, 500))
        view.run()




def main():
    hw = Homework()
    task_num = input()
    hw.run_task(task=task_num)

if __name__ == "__main__":
    main()
