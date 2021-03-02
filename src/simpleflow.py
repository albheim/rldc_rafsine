import numpy as np

class SimpleFlow:
    def __init__(self, dt, n_servers=360, n_crah=4):
        self.n_servers = n_servers
        self.n_crah = n_crah
        self.dt = dt

    def reset(self, servers, crah):
        self.server_temp_in = np.zeros(self.n_servers)
        self.server_temp_out = np.zeros(self.n_servers)
        self.crah_temp_in = np.zeros(self.n_crah)

    def step(self, servers, crah):
        # This is a step of dt and then the new values are read

        server_flow_total = np.sum(servers.flow)
        crah_flow_total = np.sum(crah.flow)

        prev_server_temp_out_avg = np.dot(servers.flow, self.server_temp_out) / server_flow_total

        bypass = min(1, crah_flow_total / server_flow_total)
        recirc = min(1, server_flow_total / crah_flow_total)

        prev_crah_temp_out = crah.temp_out[0] # Simple model has same temp out

        # All updated based on previous values
        self.server_temp_out = self.server_temp_in + servers.delta_t
        self.server_temp_in = (bypass * prev_crah_temp_out + (1 - bypass) * prev_server_temp_out_avg) * np.ones(self.n_servers)
        self.crah_temp_in = (recirc * prev_server_temp_out_avg + (1 - recirc) * prev_crah_temp_out) * np.ones(self.n_crah)