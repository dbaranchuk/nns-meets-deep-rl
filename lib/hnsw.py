from heapq import heappush, heappop, nlargest, nsmallest
from enum import Enum
import numpy as np
import torch

import multiprocessing
from .search_hnsw_swig import search_hnsw


class BaseHNSW:
    def __init__(self, graph, ef=1, k=1):
        """ Main class that handles approximate nearest neighbor search using HNSW heap-based search algorithm.
            :param graph: graph on which the search algorithm is performed
            :param ef: regulates the search algorithm "greediness"
        """
        self.graph = graph
        self.ef = ef
        self.k = k

    def get_enterpoint(self, query, **kwargs):
        vertex_id = self.get_initial_vertex_id(**kwargs)
        curdist = self.get_distance(query, self.graph.vertices[vertex_id])

        for level in range(self.graph.max_level)[::-1]:
            changed = True
            while changed:
                changed = False
                edges = list(self.graph.level_edges[vertex_id][level])
                if len(edges) == 0:
                    break

                distances = self.get_distance(query, self.graph.vertices[edges])
                for edge, dist in zip(edges, distances):
                    if dist < curdist:
                        curdist = dist
                        vertex_id = edge
                        changed = True
        return vertex_id

    def find_nearest(self, query, **kwargs):
        """
        Performs nearest neighbor lookup and returns statistics.
        :param query: vector [vertex_size] to find nearest neighbor for
        :return: nearest neighbor vertex id
        """
        if self.graph.max_level == 0:
            vertex_id = self.get_initial_vertex_id(**kwargs)
        else:
            vertex_id = self.get_enterpoint(query, **kwargs)
            self.start_session()

        visited_ids = {vertex_id}  # a set of vertices already visited by graph walker

        topResults, candidateSet = [], []
        distance = self.get_distance(query, self.graph.vertices[vertex_id])
        heappush(topResults, (-distance, vertex_id))
        heappush(candidateSet, (distance, vertex_id))
        lowerBound = distance

        while len(candidateSet) > 0:
            dist, vertex_id = heappop(candidateSet)
            if dist > lowerBound: break

            neighbor_ids = self.get_neighbors(vertex_id, visited_ids, **kwargs)
            if not len(neighbor_ids): continue

            distances = self.get_distance(query, self.graph.vertices[neighbor_ids])
            for i, (distance, neighbor_id) in enumerate(zip(distances, neighbor_ids)):
                if distance < lowerBound or len(topResults) < self.ef:
                    heappush(candidateSet, (distance, neighbor_id))
                    heappush(topResults, (-distance, neighbor_id))

                    if len(topResults) > self.ef:
                        heappop(topResults)

                    lowerBound = -nsmallest(1, topResults)[0][0]

            visited_ids.update(neighbor_ids)

        best_vertex_ids = [x[1] for x in nlargest(self.k, topResults)]
        return best_vertex_ids

    def start_session(self):
        """ Resets all logs """
        self._distance_computations = []  # number of times distance was evaluated at each step

    def get_initial_vertex_id(self, **kwargs):
        return self.graph.initial_vertex_id

    def get_neighbors(self, vertex_id, visited_ids, **kwargs):
        """ :return: a list of neighbor ids available from given vector_id. """
        neighbors = [edge for edge in self.graph.edges[vertex_id]
                     if edge not in visited_ids]
        return neighbors

    def get_distance(self, vector, vector_or_vectors):
        if len(vector_or_vectors.shape) == 1:
            self._distance_computations.append(1)
        else:
            self._distance_computations.append(vector_or_vectors.shape[0])
        return ((vector - vector_or_vectors) ** 2).sum(-1)


class HNSW(BaseHNSW):
    """
    Slow version of HNSW search algorithm that incorporates agent to predict edges. Records agent actions for training.
    This class is only needed for hierarchical graph setting which is out of the scope of our evaluation.
    If someone needs hierarchical graph support, please contact me and I'll transfer it to C++.
    In other cases use ParallelHNSW.
    """

    def record_sessions(self, agent, queries, state=None, **kwargs):
        """
        Finds nearest neighbors for several queries, computes reward and returns all that
        :param agent: lib.agent.BaseAgent
        :param queries: a batch of query vectors
            :return: a dict with a lot of metrics
        """
        if state is None:
            state = agent.prepare_state(self.graph, **kwargs)

        session_records = []
        for i, query in enumerate(queries):
            self.start_session()
            best_vertex_ids = self.find_nearest(query, agent=agent, state=state, **kwargs)
            rec = self.get_recorded_predictions()
            rec.update(
                best_vertex_id=best_vertex_ids[0],
                best_vertex_ids=best_vertex_ids,
                distance_computations=list(self._distance_computations),
                total_distance_computations=sum(self._distance_computations),
                num_hops=len(self._distance_computations) - 1,
            )
            session_records.append(rec)
        return session_records

    def start_session(self):
        """ Resets all logs and cache """
        super().start_session()
        self._edge_cache = {}  # (from_vertex_id, to_vertex_id) -> True/False

    def get_neighbors(self, vertex_id, visited_ids, agent=None, **kwargs):
        assert agent is not None, "please specify agent"

        neighbors = list(self.graph.edges[vertex_id])
        actions = agent.predict_edges(vertex_id, neighbors, **kwargs)

        chosen_neighbors = []
        for neighbor_id, action in zip(neighbors, actions):
            if neighbor_id in visited_ids: continue
            self._edge_cache[vertex_id, neighbor_id] = action
            if action: chosen_neighbors.append(neighbor_id)
        return chosen_neighbors

    def get_recorded_predictions(self):
        """ Get all the information recorded about agent actions in current session """
        history = []
        for edge, action in self._edge_cache.items():
            history.append((*edge, action))
        from_vertex_ids, to_vertex_ids, actions = zip(*history)
        return dict(from_vertex_ids=from_vertex_ids,
                    to_vertex_ids=to_vertex_ids,
                    actions=actions,)


class ServiceLabel(Enum):
    PAD = -1
    UNUSED = -2


class ParallelHNSW(BaseHNSW):
    def __init__(self, graph, ef=1, k=1, max_trajectory=80, batch_size=100000, n_jobs=1):
        """ Optimized HNSW for fast session sampling. Uses wrapped C++ code for HNSW search algorithm
            :param max_trajectory: maximum expected number of hops. Needed for swig.
                   The larger ef is, the larger max_trajectory you should set
            :param batch_size: number of edges in batch
            :param n_jobs: number of threads for C++ session sampling
        """
        super().__init__(graph, ef, k)
        assert graph.graph_type != 'hnsw', 'hnsw.ParallelHNSW does not support hierarchy. Use hnsw.HNSW.'

        self.max_trajectory = max_trajectory
        self.n_jobs = self._check_n_jobs(n_jobs)
        self.batch_size = batch_size
        self.num_edges = 0

        self.from_vertex_ids, self.to_vertex_ids, self.degrees = [], [], []
        chunk_from_vertex_ids, chunk_to_vertex_ids, chunk_degrees = [], [], []

        for vertex_id, neighbor_ids in self.graph.edges.items():
            degree = len(neighbor_ids)
            chunk_from_vertex_ids.extend([vertex_id] * degree)
            chunk_to_vertex_ids.extend(neighbor_ids)
            chunk_degrees.append(degree)
            if sum(chunk_degrees) > self.batch_size:
                self.num_edges += sum(chunk_degrees)
                self.from_vertex_ids.append(chunk_from_vertex_ids)
                self.to_vertex_ids.append(chunk_to_vertex_ids)
                self.degrees.append(chunk_degrees)
                chunk_from_vertex_ids, chunk_to_vertex_ids, chunk_degrees = [], [], []

        # Remained samples
        if len(chunk_degrees) > 0:
            self.num_edges += sum(chunk_degrees)
            self.from_vertex_ids.append(chunk_from_vertex_ids)
            self.to_vertex_ids.append(chunk_to_vertex_ids)
            self.degrees.append(chunk_degrees)

    @torch.no_grad()
    def prepare_edges_with_probs(self, agent, state, **kwargs):
        probs = np.full([state.vertices.size(0), self.graph.max_degree],
                        ServiceLabel.PAD.value, dtype=np.float32)
        edges = np.full([state.vertices.size(0), self.graph.max_degree],
                        ServiceLabel.PAD.value, dtype=np.int32)
        for i in range(len(self.from_vertex_ids)):
            edge_logp = agent.get_edge_logp(self.from_vertex_ids[i],
                                            self.to_vertex_ids[i],
                                            state=state, **kwargs)[:, 1]
            edge_probs = edge_logp.to(device='cpu').exp()
            idx = 0
            for degree in self.degrees[i]:
                vertex_id = self.from_vertex_ids[i][idx]
                probs[vertex_id, :degree] = edge_probs[idx: idx + degree]
                edges[vertex_id, :degree] = self.to_vertex_ids[i][idx: idx + degree]
                idx += degree

        return edges, probs

    def record_sessions(self, agent, queries, state=None, **kwargs):
        """
        finds nearest neighbors for several queries, computes reward and returns all that
        :param agent: lib.agent.BaseAgent
        :param queries: a batch of query vectors
        :param state: cached agent memory state. If not specified, calls agent.prepare_state
        :return: a dict with a lot of metrics
        """
        if state is None:
            state = agent.prepare_state(self.graph, **kwargs)

        num_actions = self.max_trajectory * self.graph.max_degree
        num_results = self.k + 2 + num_actions
        search_results = np.full([queries.shape[0], num_results],
                                 ServiceLabel.PAD.value, dtype=np.int32)
        edges, edge_probs = self.prepare_edges_with_probs(agent, state, **kwargs)
        trajectories = np.full([queries.shape[0], self.max_trajectory],
                               ServiceLabel.PAD.value, dtype=np.int32)
        uniform_samples = np.random.rand(queries.shape[0], num_actions).astype(np.float32)

        # search_results = [:, answer, dcs, hps]
        search_hnsw(self.graph.vertices.numpy().astype(np.float32),
                    edges, edge_probs,
                    queries.numpy().astype(np.float32),
                    trajectories, uniform_samples, search_results,
                    self.k, self.graph.initial_vertex_id,
                    self.ef, self.n_jobs)

        # Collect records
        session_records = []
        best_vertex_ids = search_results[:, :self.k]
        total_distance_computations = search_results[:, self.k]
        num_hops = search_results[:, self.k + 1]

        for i in range(queries.shape[0]):
            trajectory = trajectories[i][:num_hops[i]]
            session_actions = search_results[i, self.k + 2:]
            actions = session_actions[session_actions != ServiceLabel.PAD.value]
            actions = actions[actions != ServiceLabel.UNUSED.value]

            from_vertex_ids = np.zeros(len(actions), dtype=np.int32)
            to_vertex_ids = np.zeros(len(actions), dtype=np.int32)
            border = 0
            for from_ix_id, from_ix in enumerate(trajectory):
                vertex_actions = session_actions[from_ix_id * self.graph.max_degree:
                                                (from_ix_id + 1) * self.graph.max_degree]

                edges = np.array(self.graph.edges[from_ix], dtype=np.int32)
                vertex_actions = vertex_actions[:len(edges)]
                mask = vertex_actions != ServiceLabel.UNUSED.value
                num_samples = mask.sum()

                from_vertex_ids[border: border + num_samples] = from_ix
                to_vertex_ids[border: border + num_samples] = edges[mask]
                border += num_samples

            rec = dict(
                from_vertex_ids=from_vertex_ids.tolist(),
                to_vertex_ids=to_vertex_ids.tolist(),
                actions=actions.tolist(),
                best_vertex_id=best_vertex_ids[i][0],
                best_vertex_ids=best_vertex_ids[i],
                total_distance_computations=total_distance_computations[i],
                num_hops=num_hops[i],
            )
            session_records.append(rec)
        return session_records

    @staticmethod
    def _check_n_jobs(n_jobs):
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count() + 1 - n_jobs
        assert n_jobs > 0
        return n_jobs
