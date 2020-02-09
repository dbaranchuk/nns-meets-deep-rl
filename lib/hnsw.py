from heapq import heappush, heappop, nlargest, nsmallest
import numpy as np
import torch

import multiprocessing
from .search_hnsw_swig import search_hnsw


class HNSW:
    def __init__(self, graph, ef=1):
        """ Main class that handles approximate nearest neighbor search using HNSW heap-based search algorithm.
            :param graph: graph on which the search algorithm is performed
            :param ef: regulates the search algorithm "greediness"
        """
        self.graph = graph
        self.ef = ef

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

        best_neighbor_id = nlargest(1, topResults)[0][1]
        return best_neighbor_id

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


class ParallelHNSW(HNSW):
    def __init__(self, graph, k=1, ef=1, max_trajectory=80, batch_size=500000,
                 edge_patience=100, n_jobs=1):
        """ Optimized EdgeHNSW for fast session sampling. Uses wrapped C++ code for HNSW search algorithm
            :param max_trajectory: maximum expected number of hops. Needed for swig.
                   The larger ef is, the larger max_trajectory you should set
            :param batch_size: number of edges in batch
            :param n_jobs: number of threads for C++ session sampling
        """
        super().__init__(graph, ef)
        assert graph.graph_type != 'hnsw', 'hnsw.ParallelHNSW does not support hierarchy. Use hnsw.EdgeHNSW.'

        self.k = k
        self.max_trajectory = max_trajectory
        self.n_jobs = self._check_n_jobs(n_jobs)

        self.batch_size = batch_size
        self.edge_patience = edge_patience
        self.edge_confidence = []

        self.num_edges = 0
        self.num_confident = 0

        # Service labels to denote service fields and boundary situations
        self.service_labels = {'pad': -1, 'unused': -2, 'no_actions': -3}

        self.from_vertex_ids, self.to_vertex_ids, self.degrees = [], [], []
        chunk_from_vertex_ids, chunk_to_vertex_ids, chunk_degrees = [], [], []

        for vertex_id, neighbor_ids in self.graph.edges.items():
            degree = len(neighbor_ids)
            chunk_from_vertex_ids.extend([vertex_id] * degree)
            chunk_to_vertex_ids.extend(neighbor_ids)
            chunk_degrees.append(degree)
            if sum(chunk_degrees) > self.batch_size:
                self.num_edges += sum(chunk_degrees)
                self.from_vertex_ids.append(np.array(chunk_from_vertex_ids))
                self.to_vertex_ids.append(np.array(chunk_to_vertex_ids))
                self.degrees.append(np.array(chunk_degrees))
                self.edge_confidence.append(np.zeros(sum(chunk_degrees)))
                chunk_from_vertex_ids, chunk_to_vertex_ids, chunk_degrees = [], [], []

        # Remained samples
        if len(chunk_degrees) > 0:
            self.num_edges += sum(chunk_degrees)
            self.from_vertex_ids.append(np.array(chunk_from_vertex_ids))
            self.to_vertex_ids.append(np.array(chunk_to_vertex_ids))
            self.degrees.append(np.array(chunk_degrees))
            self.edge_confidence.append(np.zeros(sum(chunk_degrees)))

    @torch.no_grad()
    def prepare_edges_with_probs(self, agent, state=None, is_evaluate=False, greedy=False, **kwargs):
        """ :param state: cached agent memory state. If not specified, calls agent.prepare_state """
        probs = np.full([state.vertices.size(0), self.graph.max_degree], self.service_labels['pad'], dtype=np.float32)
        edges = np.full([state.vertices.size(0), self.graph.max_degree], self.service_labels['pad'], dtype=np.int32)

        if state is None:
            state = agent.prepare_state(self.graph, **kwargs)

        upper_prob_bound = 0.99989
        lower_prob_bound = 0.00011

        for i in range(len(self.from_vertex_ids)):
            edge_logp = agent.get_edge_logp(self.from_vertex_ids[i], self.to_vertex_ids[i],
                                            state=state, **kwargs).cpu()
            if greedy:
                edge_probs = edge_logp.argmax(-1).numpy()
            else:
                edge_probs = edge_logp[:, 1].exp().numpy()

            # Freeze edges that are consistently confident.
            # Set probs of confident edges to 1.1 or -0.1 to indicate the search algorithm do not sample them
            edge_probs[self.edge_confidence[i] == self.edge_patience] = 1.1
            edge_probs[self.edge_confidence[i] == -self.edge_patience] = -0.1

            if not is_evaluate:
                self.edge_confidence[i][(1. > edge_probs) & (edge_probs > upper_prob_bound)] += 1.
                self.edge_confidence[i][(0. < edge_probs) & (edge_probs < lower_prob_bound)] -= 1.
                self.edge_confidence[i][(self.edge_confidence[i] > 0) & (edge_probs < upper_prob_bound)] = 0.
                self.edge_confidence[i][(self.edge_confidence[i] < 0) & (edge_probs > lower_prob_bound)] = 0.

            idxs = np.cumsum(self.degrees[i][:-1])
            idxs = np.pad(idxs, (1, 0), 'constant', constant_values=0)
            vertex_ids = self.from_vertex_ids[i][idxs]

            mask = np.arange(self.graph.max_degree) < self.degrees[i][:, None]
            m_probs = np.full_like(mask, self.service_labels['pad'], dtype=np.float32)
            m_probs[mask] = edge_probs
            probs[vertex_ids] = m_probs

            m_edges = np.full_like(mask, self.service_labels['pad'], np.float32)
            m_edges[mask] = self.to_vertex_ids[i]
            edges[vertex_ids] = m_edges

        torch.cuda.empty_cache()
        return edges, probs

    def record_sessions(self, agent, queries, **kwargs):
        """
        finds nearest neighbors for several queries, computes reward and returns all that
        :param agent: lib.agent.BaseAgent
        :param queries: a batch of query vectors
        :return: a dict with a lot of metrics
        """
        edges, edge_probs = self.prepare_edges_with_probs(agent, **kwargs)
        num_actions = self.max_trajectory * self.graph.max_degree
        num_results = self.k + 2 + num_actions
        search_results = np.full([queries.shape[0], num_results], self.service_labels['pad'], dtype=np.int32)
        trajectories = np.full([queries.shape[0], self.max_trajectory], self.service_labels['pad'], dtype=np.int32)
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
        total_actions = search_results[:, self.k + 2:]

        for i in range(queries.shape[0]):
            trajectory = trajectories[i, :num_hops[i]]
            session_actions = total_actions[i, :num_hops[i]*self.graph.max_degree]
            session_edges = edges[trajectory].reshape(-1)
            session_mask = (session_actions != self.service_labels['pad']) & \
                           (session_actions != self.service_labels['unused'])
            actions = session_actions[session_mask]
            to_vertex_ids = session_edges[session_mask]

            idxs = np.arange(1, len(trajectory)) * self.graph.max_degree
            session_num_samples = np.array(np.array_split(session_mask, idxs)).sum(-1)
            from_vertex_ids = np.repeat(trajectory, session_num_samples)

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
