from Sim import *

class Coordinator:
    def __init__(self, configs):
        self.task_name = configs["task_name"]
        self.model_name = configs["model_name"]
        self.n_clients = configs["nclients"]
        self.p_clients = configs["pclients"]
        self.alpha = configs["alpha"]
        self.batch_size = configs["batch_size"]
        self.epoch = configs["epoch"]

        self.feature_extract, self.tokenizer, self.processor = load_pretrained(self.model_name)
        self.hd_net = load_hdnet(self.task_name, self.model_name)
        self.server = None
        self.clients = {}

        self.get_data()
        self.selection = RandomGet(self.n_clients)

    def get_data(self):
        self.test_loader, self.client_tr_loaders, self.client_te_loaders = load_client_Data(
            self.task_name, self.tokenizer, self.model_name, self.n_clients, self.alpha, self.batch_size
        )

    def main(self):
        self.server = ServerSim(self.feature_extract, self.hd_net, self.processor, self.test_loader, 0.0001, self.task_name)
        for c in range(self.n_clients):
            self.clients[c] = ClientSim(self.feature_extract, self.hd_net, self.processor, self.client_tr_loaders[c], self.client_te_loaders[c], self.batch_size)
            self.selection.register_client(c)

        for it in range(100):
            selected_ids = self.selection.select_participant(self.p_clients)
            selected_ids = sorted(selected_ids)
            global_params = self.server.get_paras()
            global_lr = self.server.get_lr()
            for client_id in selected_ids:
                self.clients[client_id].update_paras(global_params)
                self.clients[client_id].update_lr(global_lr)
                self.clients[client_id].local_training()
                client_params = self.clients[client_id].comm()
                self.server.recv_info(client_params)

            self.server.sync_paras()


if __name__ == '__main__':
    configs = {
        'task_name': "AGNEWS",
        'model_name': "DistilBERT",
        'nclients': 64,
        'pclients': 16,
        'epoch': 3,
        'batch_size': 8,
        'alpha': 0.5,
    }

    running = Coordinator(configs)
    running.main()
