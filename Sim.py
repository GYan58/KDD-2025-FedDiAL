from Functions import *
from Utils import *

class ClientSim:
    def __init__(self, pre_trained_model, hd_net, processor, train_data, test_data, batch_size=16):
        self.train_loader = cp.deepcopy(train_data)
        self.test_loader = cp.deepcopy(test_data)
        self.feature_extract = cp.deepcopy(pre_trained_model)
        self.processor = cp.deepcopy(processor)
        self.hd_net = cp.deepcopy(hd_net)
        self.lr = 0.0001
        self.fine_lr = 0.0001
        self.epoch = 3
        self.batch_size = batch_size

        self.n_class = 200

        self.pseudo_loader = None
        self.data_len = len(self.train_loader)

        self.label_data = {}
        self.labels = {}
        self.query_x = []
        self.query_y = []
        self.support_x = []
        self.support_y = []
        self._preprocess_data()

    def _preprocess_data(self):
        for batch in self.train_loader:
            features, targets = extract_features(batch, self.feature_extract, self.processor)
            inputs, targets = features.to(device), targets.to(device)
            inputs = list(inputs.cpu().detach().numpy())
            labels = list(targets.cpu().detach().numpy())

            for i in range(len(labels)):
                label = labels[i]
                if label not in self.label_data:
                    self.label_data[label] = [inputs[i]]
                    self.labels[label] = [labels[i]]
                else:
                    self.label_data[label].append(inputs[i])
                    self.labels[label].append(labels[i])

        for key in self.label_data:
            total_len = len(self.label_data[key])
            support_len = max(1, int(0.1 * total_len))
            query_len = max(1, total_len - support_len)
            left_inputs = self.label_data[key][-query_len:]
            left_labels = self.labels[key][-query_len:]
            self.label_data[key] = self.label_data[key][:support_len]
            self.labels[key] = self.labels[key][:support_len]
            self.query_x += left_inputs
            self.query_y += left_labels
            self.support_x += self.label_data[key]
            self.support_y += self.labels[key]

        self._create_loaders()

    def _create_loaders(self):
        query_inputs_tensor = torch.tensor(self.query_x).float()
        query_labels_tensor = torch.tensor(self.query_y).long()
        query_dataset = TensorDataset(query_inputs_tensor, query_labels_tensor)
        self.query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=True)

        support_inputs_tensor = torch.tensor(self.support_x).float()
        support_labels_tensor = torch.tensor(self.support_y).long()
        support_dataset = TensorDataset(support_inputs_tensor, support_labels_tensor)
        self.support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=True)

    def get_paras(self):
        return cp.deepcopy(self.hd_net.state_dict())

    def comm(self):
        return self.get_paras()

    def update_paras(self, paras):
        self.hd_net.load_state_dict(cp.deepcopy(paras))

    def update_lr(self, lr):
        self.lr = lr

    def get_lr(self):
        return self.lr

    def local_training(self):
        udf_loss = UDFLoss(num_classes=self.n_class, feat_dim=512).to(device)
        optimizer = torch.optim.AdamW(self.hd_net.parameters(), lr=self.lr, weight_decay=1e-6)

        loader = self.support_loader if self.pseudo_loader is None else self.pseudo_loader

        class_to_examples = defaultdict(list)
        for _, (inputs, targets) in enumerate(self.support_loader):
            targets = targets.to(device)
            for i in range(targets.size(0)):
                class_to_examples[targets[i].item()].append(inputs[i])

        self.hd_net.train()
        for epoch in range(self.epoch):
            for _, (features, targets) in enumerate(loader):
                inputs, targets = features.to(device), targets.to(device)
                x_ori = inputs
                x_aug = PFA(inputs)
                fea_ori = self.hd_net.dnet(x_ori)
                fea_aug = self.hd_net.dnet(x_aug)
                fea_neg = select_negative_examples_batch(self.hd_net, inputs, targets, class_to_examples)
                outputs = self.hd_net(inputs)
                loss = udf_loss(targets, outputs, fea_ori, fea_aug, fea_neg)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hd_net.parameters(), max_norm=10.0)
                optimizer.step()

                torch.cuda.empty_cache()

        self.pseudo_labeling()


    def pseudo_labeling(self):
        inputs, labels, trues = APL(self.hd_net, self.support_loader, self.query_loader).sample_and_label()

        for key in self.label_data:
            gls = [key] * len(self.label_data[key])
            gxs = self.label_data[key]
            inputs += gxs
            labels += gls
            trues += gls

        inputs = np.array(inputs)
        labels = np.array(labels)

        inputs_tensor = torch.tensor(inputs).float()
        labels_tensor = torch.tensor(labels).long()

        finetune_dataset = TensorDataset(inputs_tensor, labels_tensor)
        finetune_loader = DataLoader(finetune_dataset, batch_size=self.batch_size, shuffle=True)
        self.pseudo_loader = cp.deepcopy(finetune_loader)

    def evaluate(self, loader=None):
        self.hd_net.eval()
        loader = loader if loader is not None else self.test_loader

        correct = 0
        samples = 0
        for _, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = self.hd_net(x)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == y).sum().item()
                samples += y.shape[0]

        return samples, correct

    def one_step_finetune_evaluate(self, paras, loader=None):
        model = cp.deepcopy(self.hd_net)
        model.load_state_dict(paras)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.fine_lr, weight_decay=1e-6)
        loss_fn = nn.CrossEntropyLoss()
        bparas = cp.deepcopy(model.state_dict())

        for _, (features, targets) in enumerate(self.support_loader):
            inputs, targets = features.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            cparas = cp.deepcopy(model.state_dict())
            for key in cparas:
                if "hnet" not in key:
                    cparas[key] = cp.deepcopy(bparas[key])
            model.load_state_dict(cparas)

        model.eval()
        loader = loader if loader is not None else self.test_loader

        correct = 0
        samples = 0
        for _, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = model(x)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == y).sum().item()
                samples += y.shape[0]

        return samples, correct


class ServerSim:
    def __init__(self, pre_trained_model, hd_net, processor, test_data, lr=0.01, dname=None):
        self.lr = lr
        self.feature_extract = cp.deepcopy(pre_trained_model)
        self.model = cp.deepcopy(hd_net)
        self.processor = cp.deepcopy(processor)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98)
        self.loss_fn = nn.CrossEntropyLoss()
        self.dname = dname
        self.recv_updates = []
        self.recv_lens = []
        
        query_x = []
        query_y = []
        for batch in test_data:
            features, targets = extract_features(batch, self.feature_extract, self.processor)
            inputs, targets = features.to(device), targets.to(device)
            inputs = list(inputs.cpu().detach().numpy())
            labels = list(targets.cpu().detach().numpy())
            query_x += inputs
            query_y += labels

        query_inputs_tensor = torch.tensor(query_x).float()
        query_labels_tensor = torch.tensor(query_y).long()
        query_dataset = TensorDataset(query_inputs_tensor, query_labels_tensor)
        self.query_loader = DataLoader(query_dataset, batch_size=128, shuffle=True)

    def get_paras(self):
        return cp.deepcopy(self.model.state_dict())

    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def update_paras(self, paras):
        self.model.load_state_dict(paras)

    def recv_info(self, update, length=1):
        self.recv_updates.append(update)
        self.recv_lens.append(length)

    def avg_paras(self, paras, lens):
        res = {}
        total_len = np.sum(lens)
        for key in paras[0]:
            avg_para = sum((paras[i][key] * (lens[i] / total_len)) for i in range(len(paras)))
            res[key] = cp.deepcopy(avg_para)
        return res

    def sync_paras(self):
        avg_paras = self.avg_paras(self.recv_updates, self.recv_lens)
        self.update_paras(avg_paras)
        self.recv_updates = []
        self.recv_lens = []
        self.optimizer.step()
        self.scheduler.step()

    def evaluate(self, loader=None):
        self.model.eval()
        loader = loader if loader is not None else self.query_loader

        correct = 0
        samples = 0
        for _, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = self.model(x)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == y).sum().item()
                samples += y.shape[0]
        return samples, correct
