import torch


class BaselineInterface:
    def update(self, **kwargs):
        pass

    def get(self, **kwargs):
        pass


class MeanBaseline(BaselineInterface):
    """ Mean reward among sessions """
    def __init__(self, initial_baseline=torch.zeros(1), baseline_moving_average=0.05, temperature=20.):
        self.alpha = baseline_moving_average
        self.temperature = temperature
        self.baseline = initial_baseline
        self.step = 0

    def update(self, rewards, session_index, device='cpu', **kwargs):
        session_lengths = torch.bincount(session_index).to(torch.float32)
        mean_session_reward = torch.bincount(session_index, weights=rewards) \
                              / torch.max(session_lengths, torch.ones(*session_lengths.shape, device=device))
        mean_reward = mean_session_reward.mean().item()

        self.temperature = max(self.temperature - self.step, 1.0)
        alpha = self.temperature * self.alpha
        self.baseline = (1.0 - alpha) * self.baseline + alpha * mean_reward
        self.step += 1
        return mean_reward

    def get(self, device='cpu', **kwargs):
        return self.baseline.to(device=device)


class SessionBaseline(BaselineInterface):
    """ Collects mean reward for each session """
    def __init__(self, sessions_size, baseline_moving_average=0.05, temperature=20.):
        self.alpha = baseline_moving_average
        self.temperature = temperature
        self.baseline = torch.zeros(sessions_size)
        self.updated = torch.zeros(sessions_size, dtype=torch.uint8)
        self.step = 0

    def update(self, rewards, session_index, query_index, device='cpu', **kwargs):
        session_lengths = torch.bincount(session_index).to(torch.float32).cpu()
        padding = torch.zeros(len(query_index) - len(session_lengths))
        session_lengths = torch.cat([session_lengths, padding])

        accum_rewards = torch.cat([torch.bincount(session_index, weights=rewards).cpu(), padding])
        mean_session_reward = accum_rewards / torch.max(session_lengths, torch.ones(*session_lengths.shape))

        self.temperature = max(self.temperature - self.step, 1.0)
        # if a session has no active samples, do not change its baseline
        num_sessions = session_lengths.shape[0]
        alpha = torch.full((num_sessions,), self.temperature * self.alpha)
        alpha[session_lengths == 0] = 0
        self.baseline[query_index] = (1.0 - alpha) * self.baseline[query_index] + alpha * mean_session_reward
        self.updated[query_index] = torch.ones(1, dtype=torch.uint8)

        if self.updated.all():
            self.updated = torch.zeros_like(self.baseline, dtype=torch.uint8)
            self.step += 1

        mean_reward = mean_session_reward[session_lengths != 0].mean()
        return mean_reward.item()

    def get(self, session_index, query_index, device='cpu', **kwargs):
        return self.baseline[query_index[session_index]].to(device=device)
