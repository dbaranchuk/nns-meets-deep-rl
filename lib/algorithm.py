"""
Algorithms are training methods that
"""
import torch
import numpy as np
from .utils import compute_flat_grad, get_flat_params_from, set_flat_params_to
from torch.utils.tensorboard import SummaryWriter
from pandas import DataFrame


class BaseAlgorithm:
    """ A trainer class that updates agent parameters and draws logs """
    def __init__(self, agent, hnsw, reward, baseline=None, writer=None, device='cuda'):
        """
        :type agent: lib.agent.BaseAgent
        :type hnsw: lib.hnsw.HNSW
        :type reward: function **session_records: vector of rewards for each action in session
        :type baseline: lib.baseline.BaselineInterface
        """
        self.hnsw = hnsw
        self.agent = agent
        self.reward = reward
        self.device = device
        self.baseline = baseline
        self.writer = writer or SummaryWriter()
        self.step = 0

        self.tensor_dtypes = {
            'from_vertex_ids': torch.int64,
            'to_vertex_ids': torch.int64,
            'actions': torch.int64,
            'session_index': torch.int64,
            'rewards': torch.float32,
        }

    def get_session_batch(self, queries, ground_truth_ids, summarize=True,
                          sample_device=None, is_evaluate=False, **kwargs):
        """
        plays one session per query and ground_truth_id
        :param queries: vectors to find nearest neighbor to, [batch_size, vertex_size]
        :param ground_truth_ids: indices of actual nearest neighbors, [batch_size]
        :returns: a dict with session details
         - state: output of agent.prepare_state, common for all sessions
         - from_ix, to_ix: indices of vectors from which edge was predicted, each int32 [batch_size]
         - actions: whether edge was allowed(1) or not(0), int32 [batch_size]
         - rewards: individual rewards for each action, float32 [batch_size]
         - session_index: session index for each sample in batch, int32[batch_size]
        """
        sample_device = sample_device or self.device
        state_device = 'cpu' if is_evaluate else sample_device
        self.agent.to(device=sample_device)
        state = self.agent.prepare_state(self.hnsw.graph, device=state_device, **kwargs)
        session_records = self.hnsw.record_sessions(self.agent, queries, state=state,
                                                    **dict(kwargs, is_evaluate=is_evaluate, device=sample_device))
        self.agent.to(device=self.device)

        tensors = {name: [] for name in self.tensor_dtypes.keys()}
        for i, (query, gt, rec) in enumerate(zip(queries, ground_truth_ids, session_records)):
            rec['query'] = query
            rec['ground_truth_id'] = gt
            rec['rewards'] = self.reward(**rec)
            rec['session_index'] = [i] * len(rec['rewards'])

            if rec['actions'][0] != self.hnsw.service_labels['no_actions']:
                for col in tensors.keys():
                    tensors[col].extend(rec[col])

        tensors = {name: torch.tensor(value, dtype=self.tensor_dtypes[name], device=self.device)
                   for name, value in tensors.items()}

        results = dict(tensors, state=state)
        if summarize:
            results['summary'] = self.summarize(session_records, **kwargs)
        return results

    def summarize(self, session_records, prefix='train', write_logs=True, **kwargs):
        """ logs all information about session records """
        counters = {
            prefix + '/mean_reward': np.mean([np.mean(rec['rewards']) for rec in session_records]),
            prefix + '/recall@1': np.mean([rec['best_vertex_id'] == rec['ground_truth_id'][0]
                                           for rec in session_records]),
            prefix + '/distance_computations': np.mean([rec['total_distance_computations']
                                                        for rec in session_records]),
            prefix + '/num_hops': np.mean([rec['num_hops'] for rec in session_records]),
            prefix + '/recall@1_per_distance_computation' : \
                np.mean([int(rec['best_vertex_id'] == rec['ground_truth_id'][0]) / rec['total_distance_computations']
                         for rec in session_records])
        }
        k = len(session_records[0]['best_vertex_ids'])
        n_gt = len(session_records[0]['ground_truth_id'])
      
        if k > 1:
            assert k <= n_gt    
            recall_all = np.mean([float(len(set(rec['best_vertex_ids']) & set(rec['ground_truth_id'][:k].tolist()))) / k 
                                  for rec in session_records])
            counters[prefix + '/recall@%i' % k] = recall_all
            counters[prefix + '/recall@%i_per_distance_computation' % k] = \
                np.mean([float(len(set(rec['best_vertex_ids']) & set(rec['ground_truth_id'][:k].tolist()))) / (rec['total_distance_computations'] * k)
                         for rec in session_records])
        if write_logs:
            for key, value in counters.items():
                self.writer.add_scalar(key, value, global_step=self.step)
        return counters

    def train_step(self, batch_queries, batch_ground_truth_ids, **kwargs):
        """ samples sessions and performs update step
        :param batch_queries: vectors to find nearest neighbor to, [batch_size, vertex_size]
        :param batch_ground_truth_ids: indices of actual nearest neighbors, [batch_size]
        :returns: mean reward
        """
        batch_records = self.get_session_batch(batch_queries, batch_ground_truth_ids, **kwargs)
        mean_reward = self.train_on_batch(**batch_records, **kwargs)
        self.step += 1
        return mean_reward

    def train_on_batch(self, **rec_kwargs):
        """ updates agent parameters on sampled sessions"""
        raise NotImplementedError()

    def evaluate(self, batch_queries, batch_ground_truth_ids, prefix='dev', **kwargs):
        """ Compute metrics and write logs for current agent state
        :param batch_queries: vectors to find nearest neighbor to, [batch_size, vertex_size]
        :param batch_ground_truth_ids: indices of actual nearest neighbors, [batch_size]
        :param prefix: prefix for metric names
        """
        summary = self.get_session_batch(batch_queries, batch_ground_truth_ids, greedy=True,
                                   summarize=True, prefix=prefix, is_evaluate=True, **kwargs)['summary']
        mean_reward = np.mean(summary[prefix + '/mean_reward'])
        return mean_reward

    @staticmethod
    def aggregate_samples(from_vertex_ids, to_vertex_ids, actions, advantages, device='cuda'):
        """ Merge the same samples """
        df = DataFrame({'from_vertex_ids': from_vertex_ids.cpu(), 'to_vertex_ids': to_vertex_ids.cpu(),
                        'actions': actions.cpu(), 'advantages': advantages.cpu(),
                        'freqs': torch.ones(len(actions)).type(torch.float32)})
        df = df.groupby(['from_vertex_ids', 'to_vertex_ids', 'actions'], sort=False).sum().reset_index()

        # Use of Gumbel Max Trick for unbiased Fvp estimate when train on subset of samples
        df['probs'] = np.log(df['freqs']) + np.random.gumbel(0, 1, len(df))
        df = df.sort_values(by=['probs'], ascending=False)
        del df['probs']
        return [torch.tensor(df[column].values, device=device) for column in df.columns]


class TRPO(BaseAlgorithm):
    """ Trust Region Policy Optimization, see https://arxiv.org/pdf/1502.05477.pdf.
        Deprecated, use EfficientTRPO instead
    """
    def __init__(self, agent, hnsw, reward, baseline, max_kl=0.01, damping=0.1, entropy_reg=0.0, **kwargs):
        super().__init__(agent, hnsw, reward, baseline, **kwargs)
        self.max_kl = max_kl
        self.damping = damping
        self.entropy_reg = entropy_reg

    def linesearch(self, f, x, fullstep):
        max_backtracks = 10
        loss, _, _ = f(x)
        powers = torch.arange(max_backtracks, dtype=torch.float32).cuda()
        for stepfrac in .5 ** powers:
            xnew = x + stepfrac * fullstep
            new_loss, kl, _ = f(xnew)
            actual_improve = new_loss - loss
            if kl.item() <= self.max_kl and actual_improve.item() < 0:
                x = xnew
                loss = new_loss
        return x

    def conjugate_gradient(self, f_Ax, b, cg_iters=10, residual_tol=1e-10):
        p = b.clone()
        r = b.clone()
        x = torch.zeros(b.size()).cuda()
        rdotr = torch.sum(r * r)
        for i in range(cg_iters):
            z = f_Ax(p)
            v = rdotr / (torch.sum(p * z) + 1e-8)
            x += v * p
            r -= v * z
            newrdotr = torch.sum(r * r)
            mu = newrdotr / (rdotr + 1e-8)
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < residual_tol:
                break
        return x

    def train_on_batch(self, state, from_vertex_ids, to_vertex_ids, actions, rewards, session_index, **kwargs):
        baseline = self.baseline.get(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                     rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        advantage = rewards - baseline

        # Update baseline for the next iteration
        mean_reward = self.baseline.update(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                           rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        # if not train mode, exit
        if self.max_kl == 0.0:
            return mean_reward

        # Aggregate samples
        from_vertex_ids, to_vertex_ids, actions, advantage, freqs = \
            self.aggregate_samples(from_vertex_ids, to_vertex_ids, actions, advantage, device=self.device)

        state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)
        logp = self.agent.get_edge_logp(from_vertex_ids, to_vertex_ids, state=state, device=self.device)
        logp_action = torch.gather(logp, dim=-1, index=actions[:, None])[:, 0]

        old_logp = logp.detach()
        old_logp_action = logp_action.detach()

        ratio = torch.exp(logp_action - old_logp_action)  # pi(a|s) / pi_old(a|s)
        loss = -(ratio * advantage).sum() / freqs.sum()

        grads = torch.autograd.grad(loss, self.agent.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach_()

        def Fvp(v):
            # Here we compute Fx to do solve Fx = g using conjugate gradients
            state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)
            logp = self.agent.get_edge_logp(from_vertex_ids, to_vertex_ids, state=state, device=self.device)
            probs = logp.exp()
            kl = (freqs * (probs * (logp - old_logp)).sum(-1)).sum() / freqs.sum()
            assert (kl > -0.0001).all() and (kl < 10000).all()

            grads = torch.autograd.grad(kl, self.agent.parameters(), create_graph=True)

            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, self.agent.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach_()
            return flat_grad_grad_kl + v * self.damping

        stepdir = self.conjugate_gradient(Fvp, -loss_grad, 10)

        # Here we compute the initial vector to do linear search
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm[0]

        # Here we get the start point
        prev_params = get_flat_params_from(self.agent)

        @torch.no_grad()
        def get_loss_kl_ent(params):
            # Helper for linear search
            set_flat_params_to(self.agent, params)
            state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)

            logp = self.agent.get_edge_logp(from_vertex_ids, to_vertex_ids, state=state, device=self.device)
            logp_action = torch.gather(logp, dim=-1, index=actions[:, None])[:, 0]
            probs = torch.exp(logp)
            kl = (freqs * (probs * (logp - old_logp)).sum(-1)).sum() / freqs.sum()

            ratio = torch.exp(logp_action - old_logp_action)  # pi(a|s) / pi_old(a|s)
            loss = -(ratio * advantage).sum() / freqs.sum()
            ent = (freqs * (-probs * logp).sum(-1)).sum() / freqs.sum()
            assert (kl > -0.0001).all() and (kl < 10000).all()
            return [loss, kl, ent]

        # Here we find our new parameters
        new_params = self.linesearch(get_loss_kl_ent, prev_params, fullstep)
        del state  # state becomes obsolete at this point

        # And we set it to our network
        set_flat_params_to(self.agent, new_params)

        # Summarize
        loss, kl, ent = get_loss_kl_ent(new_params)
        self.writer.add_scalar('train/baseline', baseline.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/advantage', advantage.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/entropy', ent.item(), global_step=self.step)
        self.writer.add_scalar('train/kl', kl.item(), global_step=self.step)
        self.writer.add_scalar('train/loss', loss.item(), global_step=self.step)
        return mean_reward


class EfficientTRPO(TRPO):
    """ Optimized Trust Region Policy Optimization """

    def __init__(self, agent, hnsw, reward, baseline, samples_in_batch=100000,
                 Fvp_speedup=5, Fvp_type='fim', Fvp_min_batches=10, **kwargs):
        super().__init__(agent, hnsw, reward, baseline, **kwargs)
        self.samples_in_batch = samples_in_batch
        self.Fvp_min_batches = Fvp_min_batches
        self.Fvp_speedup = Fvp_speedup
        self.Fvp_type = Fvp_type

    def train_on_batch(self, state, from_vertex_ids, to_vertex_ids, actions, rewards, session_index, **kwargs):
        baseline = self.baseline.get(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                     rewards=rewards, session_index=session_index, device=self.device, **kwargs)

        # Update baseline for the next iteration
        mean_reward = self.baseline.update(state=state, from_vertex_ids=from_vertex_ids, to_vertex_ids=to_vertex_ids,
                                           rewards=rewards, session_index=session_index, device=self.device, **kwargs)
        # if not train mode, exit
        if self.max_kl == 0.0:
            return mean_reward

        advantage = rewards - baseline

        from_vertex_ids, to_vertex_ids, actions, advantage, freqs = \
            self.aggregate_samples(from_vertex_ids, to_vertex_ids, actions, advantage, device=self.device)

        batches_from_vertex_ids = from_vertex_ids.split(self.samples_in_batch)
        batches_to_vertex_ids = to_vertex_ids.split(self.samples_in_batch)
        batches_advantage = advantage.split(self.samples_in_batch)
        batches_actions = actions.split(self.samples_in_batch)
        batches_freqs = freqs.split(self.samples_in_batch)
        batches_old_logp = []

        loss_grad = 0
        for batch_from_vertex_ids, batch_to_vertex_ids, batch_actions, batch_advantage, batch_freqs in \
                zip(batches_from_vertex_ids, batches_to_vertex_ids, batches_actions, batches_advantage, batches_freqs):
            batch_logp = self.agent.get_edge_logp(batch_from_vertex_ids, batch_to_vertex_ids,
                                                  state=state, device=self.device)
            batch_logp_actions = torch.gather(batch_logp, dim=-1, index=batch_actions[:, None])[:, 0]
            batch_old_logp = batch_logp.detach()
            batch_old_logp_actions = batch_logp_actions.detach()

            batches_old_logp.append(batch_old_logp)

            # Here ratio is always 1
            ratio = torch.exp(batch_logp_actions - batch_old_logp_actions)  # pi(a|s) / pi_old(a|s)
            loss = -(ratio * batch_advantage).sum() / freqs.sum()
            # Entropy requlizer
            ent = (-batch_logp.exp() * batch_logp).sum(-1)
            loss -= self.entropy_reg * (batch_freqs * ent).sum() / freqs.sum()  # add entropy regularization

            grads = torch.autograd.grad(loss, self.agent.parameters())
            loss_grad += torch.cat([grad.view(-1) for grad in grads]).detach_()

        # Forward Fisher vector product
        def Fvp_forward(v):
            # Here we compute Fx to do solve Fx = g using conjugate gradients
            flat_grad_grad_kl = 0
            min_n_batches = min(self.Fvp_min_batches, len(batches_freqs))
            nbatches = max(len(batches_freqs) // self.Fvp_speedup, min_n_batches)

            for i_batch, (batch_from_vertex_ids, batch_to_vertex_ids, batch_old_logp, batch_freqs) in \
                enumerate(zip(batches_from_vertex_ids, batches_to_vertex_ids, batches_old_logp, batches_freqs)):
                if i_batch == nbatches: break
                batch_logp = self.agent.get_edge_logp(batch_from_vertex_ids, batch_to_vertex_ids,
                                                      state=state, device=self.device)
                batch_probs = batch_logp.exp()
                kl = (batch_freqs * (batch_probs * (batch_logp - batch_old_logp)).sum(-1)).sum()
                assert (kl > -0.0001).all()

                flat_grad_kl = compute_flat_grad(kl, self.agent.parameters(), create_graph=True)

                kl_v = (flat_grad_kl * v).sum()
                flat_grad_grad_kl += compute_flat_grad(kl_v, self.agent.parameters()).detach_()
            flat_grad_grad_kl /= sum([batch_freqs.sum() for batch_freqs in batches_freqs[:nbatches]])
            return flat_grad_grad_kl + v * self.damping

        # Fisher vector product Requires ~15% less memory
        def Fvp_fim(v):
            JTMJv = 0
            min_n_batches = min(self.Fvp_min_batches, len(batches_freqs))
            nbatches = max(len(batches_freqs) // self.Fvp_speedup, min_n_batches)

            for (batch_from_vertex_ids, batch_to_vertex_ids, batch_freqs) in \
                zip(batches_from_vertex_ids[:nbatches], batches_to_vertex_ids[:nbatches], batches_freqs[:nbatches]):
                batch_probs = self.agent.get_edge_logp(batch_from_vertex_ids, batch_to_vertex_ids,
                                                       state=state, device=self.device).exp()
                mu = batch_probs.view(-1)  # mu = mu.view(-1)
                M = mu.pow(-1).detach()
                weighted_mu = (batch_freqs[:, None] * batch_probs).view(-1)
                # M is the second derivative of the KL distance wrt network output
                # (M*M diagonal matrix compressed into a M*1 vector)
                # mu is the network output (M*1 vector)
                t = torch.ones(mu.size(), device=self.device, requires_grad=True)
                mu_t = mu @ t
                Jt = compute_flat_grad(mu_t, self.agent.parameters(), create_graph=True)
                Jtv = Jt @ v
                Jv = torch.autograd.grad(Jtv, t)[0]
                mu_MJv = (weighted_mu * M * Jv.detach_()).sum()
                JTMJv += compute_flat_grad(mu_MJv, self.agent.parameters()).detach_()
            sum_freqs = sum([batch_freqs.sum() for batch_freqs in batches_freqs[:nbatches]])
            JTMJv /= sum_freqs
            return JTMJv + v * self.damping

        Fvp = Fvp_fim if self.Fvp_type == 'fim' else Fvp_forward
        stepdir = self.conjugate_gradient(Fvp, -loss_grad, 10)

        # Here we compute the initial vector to do linear search
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm[0]

        # Here we get the start point
        prev_params = get_flat_params_from(self.agent)

        @torch.no_grad()
        def get_loss_kl_ent(params):
            # Helper for linear search
            set_flat_params_to(self.agent, params)

            loss, kl, ent = 0, 0, 0
            for batch_from_vertex_ids, batch_to_vertex_ids, batch_actions, \
                batch_advantage, batch_old_logp, batch_freqs in \
                    zip(batches_from_vertex_ids, batches_to_vertex_ids, batches_actions,
                        batches_advantage, batches_old_logp, batches_freqs):
                batch_logp = self.agent.get_edge_logp(batch_from_vertex_ids, batch_to_vertex_ids,
                                                      state=state, device=self.device)
                batch_probs = batch_logp.exp()
                batch_logp_action = torch.gather(batch_logp, dim=-1, index=batch_actions[:, None])[:, 0]
                batch_old_logp_action = torch.gather(batch_old_logp, dim=-1, index=batch_actions[:, None])[:, 0]
                ratio = torch.exp(batch_logp_action - batch_old_logp_action)

                loss += -torch.sum(ratio * batch_advantage)
                kl += (batch_freqs * (batch_probs * (batch_logp - batch_old_logp)).sum(-1)).sum()
                ent += (batch_freqs * (-batch_probs * batch_logp).sum(-1)).sum()
            n_samples = freqs.sum()
            loss /= n_samples
            kl /= n_samples
            ent /= n_samples
            assert (kl > -0.0001).all() and (kl < 10000).all()
            return [loss - self.entropy_reg * ent, kl, ent]

        # Here we find our new parameters
        state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)
        new_params = self.linesearch(get_loss_kl_ent, prev_params, fullstep)

        # And we set it to our network
        set_flat_params_to(self.agent, new_params)

        # Summarize
        loss, kl, ent = get_loss_kl_ent(new_params)
        self.writer.add_scalar('train/baseline', baseline.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/advantage', advantage.mean().item(), global_step=self.step)
        self.writer.add_scalar('train/entropy', ent.item(), global_step=self.step)
        self.writer.add_scalar('train/kl', kl.item(), global_step=self.step)
        self.writer.add_scalar('train/loss', loss.item(), global_step=self.step)
        return mean_reward
