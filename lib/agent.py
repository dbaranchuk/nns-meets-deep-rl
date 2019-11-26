import torch
import torch.nn as nn
from collections import namedtuple


class BaseAgent(nn.Module):
    # State is arbitrary information that agent needs to compute about its graph. Immutable.
    State = namedtuple("AgentState", ['vertices'])

    def prepare_state(self, graph, **kwargs):
        """ Pre-computes graph representation for further use in edge prediction """
        return self.State(vertices=graph.vertices)

    def predict_edges(self, vertex_id, neighbor_ids, state, **kwargs):
        """
        For each node in neighbor_ids, predicts whether it is available from vertex_idx
        :param vertex_id: vertex index (0-based)
        :param neighbor_ids: neighbor vertex indices, a list [num_neighbors]
        :param state: output of prepare_state function
        :return: 0/1 vector for each neighbor in neighbor ids
            1 if agent allows an edge between vertex_id and and that neighbor,
            0 if there is no edge
        """
        return [1] * len(neighbor_ids)


class ProbabilisticAgent(BaseAgent):
    """ Agent with:
        * `get_edge_logp` method which should be implemented in your successor class.
        * `predict_edges` which uses edges' logprobs to predict them.
            Normally, you shouldn't override it.
    """

    # State is arbitrary information that agent needs to compute about its graph. Immutable.
    State = namedtuple("AgentState", ['vertices', 'logp_cache'])

    def prepare_state(self, graph, **kwargs):
        """ Pre-computes graph representation for further use in edge prediction """
        return self.State(vertices=graph.vertices, logp_cache={})

    def get_edge_logp(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu', **kwargs):
        """ Take vertices and predict probability of an edge between them. """
        raise NotImplementedError

    def predict_edges(self, vertex_id, neighbor_ids, greedy=False, state=None, **kwargs):
        """
        For each node in neighbor_ids, predicts whether it is available from vertex_idx
        :param vertex_id: vertex index (0-based)
        :param neighbor_ids: neighbor vertex indices, a list [num_neighbors]
        :param logp_cache: precomputed logp for vertices in session
        :param greedy: whether to take argmax or sample according to probs
        :return: 0/1 vector for each neighbor in neighbor ids
            1 if agent allows an edge between vertex_id and and that neighbor,
            0 if there is no edge
        """
        with torch.no_grad():
            if vertex_id not in state.logp_cache.keys():
                edge_logp = self.get_edge_logp([vertex_id] * len(neighbor_ids), neighbor_ids, state=state, **kwargs)
                edge_logp = edge_logp.to(device='cpu')
                state.logp_cache[vertex_id] = edge_logp
            else:
                edge_logp = state.logp_cache[vertex_id]

            if greedy:
                return edge_logp.argmax(dim=-1)
            else:
                return torch.multinomial(torch.exp(edge_logp), 1)[:, 0]


class SimpleNeuralAgent(ProbabilisticAgent):
    """ Agent with a feedwforward neural network for edge prediction """
    def __init__(self, vertex_size, hidden_size, activation=nn.ELU(), min_prob=1e-4):
        super().__init__()

        self.min_prob, self.max_prob = min_prob, 1. - min_prob
        self.edge_network = nn.Sequential(
            nn.Linear(2 * vertex_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, 1),
        )

    def get_edge_logp(self, from_vertex_ids, to_vertex_ids, *, state, device='cpu', **kwargs):
        """
        :param from_vertex_ids: indices of vertices from which there could be and edge, [batch_size]
        :param to_vertex_ids: indices of vertices to which there could be an edge, [batch_size]
        :return: log-probabilities of taking and not taking edge between vertex1 and vertex2,
            shape: [batch_size, 2]
        """

        vertices_from = state.vertices[from_vertex_ids, :].to(device=device)
        vertices_to = state.vertices[to_vertex_ids, :].to(device=device)

        nn_inputs = torch.cat([vertices_from, vertices_to], dim=-1)
        theta = torch.sigmoid(self.edge_network(nn_inputs))
        theta = theta * (self.max_prob - self.min_prob) + self.min_prob
        probs = torch.cat([theta, 1. - theta], dim=-1)
        return probs.log()

