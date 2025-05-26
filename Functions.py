from Utils import *

class RandomGet:
    def __init__(self, Nclients=0):
        self.totalArms = OrderedDict()
        self.IDsPool = []
        self.Round = 0
        self.Clients = Nclients

    def register_client(self, clientId):
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['status'] = True

    def updateStatus(self, Id, Sta):
        self.totalArms[Id]['status'] = Sta

    def select_participant(self, num_of_clients):
        viable_clients = [x for x in self.totalArms.keys() if self.totalArms[x]['status']]
        return self.getTopK(num_of_clients, viable_clients)

    def getTopK(self, numOfSamples, feasible_clients):
        IDs = []
        for i in range(len(feasible_clients)):
            IDs.append(i)
        rd.shuffle(IDs)
        pickedClients = IDs[:numOfSamples]
        return pickedClients

def add_noise(data, noise_level=0.01):
    noise = noise_level * torch.randn_like(data)
    return data + noise

def random_masking(data, mask_prob=0.5):
    mask = torch.rand_like(data) < mask_prob
    return data.masked_fill(mask, 0)

def random_scaling(data, scale_range=(0.9, 1.1)):
    a = scale_range[0]
    b = scale_range[1]
    scale = (b - a) * torch.rand_like(data) + a
    return data * scale.to(data.device)

def PFA(data):
    Prob = np.random.rand()
    if Prob <= 1/3:
        return add_noise(data)
    if Prob > 1/3 and Prob <= 2/3:
        return random_masking(data)
    if Prob > 2/3:
        return random_scaling(data)

class UDFLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(UDFLoss, self).__init__()
        self.lamba = 1.0
        self.gamma = 0.001

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = Parameter(torch.randn(num_classes, feat_dim))

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_ori, y_pred, x, x_pos, x_neg):
        loss = self.ce_loss(y_pred, y_ori)
        loss += self.triplet_loss(x, x_pos, x_neg) * self.lamba
        batch_size = x.size(0)

        x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
        centers_sq = torch.sum(self.centers ** 2, dim=1, keepdim=True)
        distmat = x_sq + centers_sq.T - 2 * torch.matmul(x, self.centers.T)

        classes = torch.arange(self.num_classes, device=x.device).long()
        labels = y_ori.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss += dist.clamp(min=1e-12, max=1e+12).sum() / batch_size * self.gamma

        return loss


def select_negative_examples_batch(model, inputs, targets, class_to_examples):
    device = inputs.device
    batch_size = inputs.size(0)
    fea_neg_batch = torch.zeros_like(inputs)

    for i in range(batch_size):
        label = targets[i].item()
        all_labels = list(class_to_examples.keys())
        if label in all_labels:
            all_labels.remove(label)
        negative_samples = []
        negative_labels = []
        for neg_label in all_labels:
            negative_samples.extend(class_to_examples[neg_label])
            negative_labels.extend([neg_label] * len(class_to_examples[neg_label]))
        negative_samples = torch.stack(negative_samples).to(device)
        input_sample = inputs[i].unsqueeze(0)
        similarities = F.cosine_similarity(input_sample, negative_samples)
        min_sim_index = torch.argmin(similarities)
        fea_neg_batch[i] = negative_samples[min_sim_index]

    fea_outputs = model.dnet(fea_neg_batch)
    return fea_outputs.squeeze()


class APL:
    def __init__(self, model, labeled_data_loader, unlabeled_data_loader):
        self.model = cp.deepcopy(model)
        self.labeled_data_loader = labeled_data_loader
        self.unlabeled_data_loader = unlabeled_data_loader

    def compute_embeddings(self, data_loader, return_labels=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        datas = []
        embeddings = []
        all_labels = []
        preds = []
        for data, labels in data_loader:
            data = data.to(device)
            datas += list(data.cpu().detach().numpy())
            with torch.no_grad():
                features = self.model.dnet(data)
                embeddings += list(features.cpu().numpy())
                outputs = self.model(data)
                _, pseudo_labels = torch.max(outputs.data, 1)
                preds += list(pseudo_labels.cpu().numpy())
                if return_labels:
                    all_labels += list(labels.cpu().numpy())

        if return_labels:
            return datas, embeddings, all_labels, preds
        return datas, embeddings, preds

    def calculate_prototypes(self, embeddings, labels):
        labels = torch.tensor(labels)
        unique_labels = torch.unique(labels)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        prototypes = torch.stack([embeddings[labels == label].mean(0) for label in unique_labels])
        return prototypes, unique_labels

    def assign_pseudo_labels(self):
        datas, embeddings, labels, _ = self.compute_embeddings(self.labeled_data_loader, return_labels=True)
        prototypes, prototype_labels = self.calculate_prototypes(embeddings, labels)

        query_datas, query_embeddings, query_trues, query_pseudos = self.compute_embeddings(self.unlabeled_data_loader,return_labels=True)

        query_embeddings = torch.tensor(query_embeddings, dtype=torch.float32)
        dists = torch.cdist(query_embeddings, prototypes)

        probs = F.softmax(-dists, dim=1)
        pseudo_labels = torch.argmax(probs, dim=1)

        mapped_labels = prototype_labels[pseudo_labels]
        mapped_labels = mapped_labels.cpu().numpy()
        probs = probs.cpu().numpy()
        probs = [p for i, p in enumerate(probs.max(1))]

        filter_labels = []
        filter_datas = []
        filter_trues = []
        filter_probs = []
        for i in range(len(mapped_labels)):
            if mapped_labels[i] == query_pseudos[i]:
                filter_datas.append(query_datas[i])
                filter_labels.append(mapped_labels[i])
                filter_trues.append(query_trues[i])
                filter_probs.append(probs[i])
        return filter_datas, filter_labels, filter_probs, filter_trues

    def sample_and_label(self):
        datas, pseudo_labels, probs, trues = self.assign_pseudo_labels()

        if len(datas) == 0:
            return [], [], []

        confidence_threshold = np.percentile(probs, 75)
        high_confidence_indices = [i for i in range(len(probs)) if probs[i] >= confidence_threshold]
        final_data = [datas[i] for i in high_confidence_indices]
        final_labels = [pseudo_labels[i] for i in high_confidence_indices]
        final_trues = [trues[i] for i in high_confidence_indices]

        return final_data, final_labels, final_trues