import copy
import numpy as np
from collections import OrderedDict, defaultdict


class MicrobiomeTree(object):

    class Node(object):
        def __init__(self, name):
            self.name = name

            self.edge_wt = None  # length of the edge to this node
            self.parent = []
            self.children = []

            self.distance_to_root = None
            self.level = None
            self.descendent = []
            self.antecedent = []

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.nodes = dict()
        self.root_name = None

    def add_node(self, name):
        self.nodes[name] = self.Node(name)

    def add_edge(self, parent_name, children_name, edge_wt):
        self.nodes[parent_name].children.append(children_name)
        self.nodes[children_name].parent.append(parent_name)
        self.nodes[children_name].edge_wt = edge_wt

    def _dfs_preprocess(self, name, visited, path):
        path.append(name)
        visited.add(name)
        visited_original = visited.copy()
        node = self.nodes[name]
        for next_name in node.children:
            if next_name not in visited:
                self.nodes[next_name].distance_to_root = self.nodes[next_name].edge_wt + node.distance_to_root
                self._dfs_preprocess(next_name, visited, path)
        path.pop()
        node.level = len(path)
        node.descendent.extend(list(visited - visited_original))
        node.antecedent.extend(path)

    def preprocess(self):
        for _, node in self.nodes.items():
            if len(node.parent) == 0:
                assert self.root_name is None
                self.root_name = node.name
        assert self.root_name is not None

        self.nodes[self.root_name].distance_to_root = 0

        path = []
        visited = set()
        self._dfs_preprocess(self.root_name, visited, path)

    def leaves(self):
        return [node.name for _, node in self.nodes.items() if len(node.children) == 0]

    def _rename_shrink(self, old_name, new_name, new_distance_to_root):
        self.nodes[new_name] = copy.deepcopy(self.nodes[old_name])
        self.nodes[new_name].name = new_name
        self.nodes[new_name].distance_to_root = new_distance_to_root
        self.nodes[new_name].edge_wt = new_distance_to_root - (
                self.nodes[old_name].distance_to_root - self.nodes[old_name].edge_wt)
        self.nodes[new_name].children = []
        self.nodes[new_name].descendent = []
        del self.nodes[old_name]

    def _remove_and_modify(self, remove, modify):
        nodes_copy = copy.deepcopy(self.nodes)
        for node_name, node in nodes_copy.items():
            if node_name in remove:
                del self.nodes[node_name]
            elif node_name in modify:
                self._rename_shrink(node_name, modify[node_name][0], modify[node_name][1])
            else:
                self.nodes[node_name].children = [i if i not in modify else modify[i][0]
                                                  for i in self.nodes[node_name].children if i not in remove]
                self.nodes[node_name].descendent = [i if i not in modify else modify[i][0]
                                                    for i in self.nodes[node_name].descendent if i not in remove]

    def _dfs_get_partition(self, name, visited, result, remove, modify, thres, next_thres):
        visited.add(name)
        node = self.nodes[name]
        use_pseudo = False if (node.distance_to_root == 0 or next_thres <= node.distance_to_root < thres) and \
            all(self.nodes[child].distance_to_root >= thres for child in node.children) else True
        for next_name in node.children:
            if next_name not in visited:
                if self.nodes[next_name].distance_to_root >= thres:
                    if use_pseudo:
                        current_remove = self.nodes[next_name].descendent
                        result[next_name + '*'].extend(current_remove + [next_name])
                        modify[next_name] = (next_name + '*', (thres + next_thres) / 2)
                        remove.extend(current_remove)
                    else:
                        current_remove = self.nodes[next_name].descendent + [next_name]
                        result[name].extend(current_remove)
                        remove.extend(current_remove)
                else:
                    self._dfs_get_partition(next_name, visited, result, remove, modify, thres, next_thres)

    def get_partition(self, bandwidth, verbose=True):
        tree_original = copy.deepcopy(self.__dict__)
        max_distance_to_root = max(node.distance_to_root for _, node in self.nodes.items())
        all_thres = list(np.arange(max(0, max_distance_to_root - bandwidth), 0, -bandwidth).round(4)) + [0, -1]
        all_result = list()
        for i in range(len(all_thres) - 1):
            visited = set()
            result = defaultdict(list)
            remove = list()
            modify = dict()
            self._dfs_get_partition(self.root_name, visited, result, remove, modify, all_thres[i], all_thres[i + 1])
            self._remove_and_modify(remove, modify)
            all_result.append(dict(result))
        self.__dict__ = tree_original

        partition = dict(zip(all_thres[:-1], all_result))
        partition_flatten_list = list()
        for _, v1 in partition.items():
            for k2, v2 in v1.items():
                v2.sort(key=lambda x: int(x.strip('*')) + x.count('*') * (10 ** len(str(len(self.nodes)))))
                partition_flatten_list.append((k2, v2))
        partition_flatten = OrderedDict(partition_flatten_list)

        '''
        The partition_flatten_cleaned is a dictionary, each key of which is a node name, 
        each value of which is a nonempty list of node names. It also satisfies:
        (1) All its keys form the set of output nodes; all elements in its values form the set of input nodes, which is 
            a union of two disjoint sets: 1) the set of output nodes and 2) the set of leaf nodes
        (2) It is an ordered dictionary, where the last key is root_name, and partition_flatten_cleaned[k] is a subset 
            of the union of two disjoint sets: 1) the set of keys before k and 2) the set of leaf nodes 
            (i.e., it doesn't contain any keys after k)
        '''
        output_nodes_set = set(partition_flatten.keys())
        leaves_set = set(self.leaves())
        partition_flatten_cleaned = {k: [i for i in v if i in output_nodes_set | leaves_set]
                                     for k, v in partition_flatten.items()}
        assert len(output_nodes_set & leaves_set) == 0
        assert all(len(v) > 0 for _, v in partition_flatten_cleaned.items())
        assert list(partition_flatten_cleaned.keys())[-1] == self.root_name
        keys_encountered = set()
        for k, v in partition_flatten_cleaned.items():
            assert set(v).issubset(keys_encountered | leaves_set)
            keys_encountered.add(k)

        if verbose:
            print('max distance to root is {0:} and all thresholds are {1:}'.format(
                max_distance_to_root, ', '.join([str(i) for i in all_thres[:-1]])))
            for k1, v1 in partition.items():
                print(k1, v1)
            print(partition_flatten_cleaned)

        return partition, partition_flatten_cleaned
