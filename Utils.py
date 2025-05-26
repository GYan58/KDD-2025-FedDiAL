from Settings import *

def load_pretrained(model_name):
    model = None
    tokenizer = None
    processor = None

    if model_name == "ResNet-152":
        model = models.resnet152(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Identity()

    elif model_name == "GPT-2":
        model = GPT2Model.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

    elif model_name == "DistilBERT":
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model.eval()

    elif model_name == "CLIP":
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        model.eval()

    return model, tokenizer, processor

def extract_features(batch, model, processor=None):
    with torch.no_grad():
        if isinstance(model, (GPT2Model, DistilBertModel)):
            tokens, batch_labels = batch
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)
            batch_features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(model, CLIPModel):
            images, batch_labels = batch
            pil_images = [transforms.ToPILImage()(img) for img in images]
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.get_image_features(**inputs)
            batch_features = outputs
        else:
            images, batch_labels = batch
            images = images.to(device)
            outputs = model(images)
            batch_features = outputs

        return batch_features, batch_labels

class HDNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HDNet, self).__init__()
        self.d_net = nn.Linear(input_dim, 512)
        self.hnet = nn.Linear(512, num_classes)
        self.to(device)

    def dnet(self, x):
        return self.d_net(x)

    def forward(self, x):
        x = self.dnet(x)
        x = self.hnet(x)
        return x

def load_hdnet(task_name, model_name=None):
    if task_name in ["CIFAR-100", "Tiny ImageNet"]:
        if model_name == "ResNet-152":
            input_dim = 2048
        if model_name == "CLIP":
            input_dim = 512
    elif task_name in ["AGNEWS", "YELP-F"]:
        input_dim = 768
    else:
        raise ValueError("Unsupported task name")

    if task_name == "CIFAR-100":
        num_classes = 100
    elif task_name == "Tiny ImageNet":
        num_classes = 200
    elif task_name == "AGNEWS":
        num_classes = 4
    elif task_name == "YELP-F":
        num_classes = 5
    else:
        raise ValueError("Unsupported task name")

    hd_net = HDNet(input_dim, num_classes)
    return hd_net

def collate_fn(batch, tokenizer, model_name):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    tokens = None
    if model_name == "DistilBERT":
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    if model_name == "GPT-2":
        tokens = tokenizer(texts, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt", max_length=1024)
    labels = torch.tensor(labels)
    return tokens, labels

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        text, label = item['text'], item['label']
        return {'text': text, 'label': label}

def load_specific_dataset(task_name):
    if task_name == "CIFAR-100":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    elif task_name == "Tiny ImageNet":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        test_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)

    elif task_name == "AGNEWS":
        dataset = load_dataset('ag_news')
        train_dataset = CustomDataset(dataset['train'])
        test_dataset = CustomDataset(dataset['test'])

    elif task_name == "YELP-F":
        dataset = load_dataset('yelp_review_full')
        train_dataset = CustomDataset(dataset['train'])
        test_dataset = CustomDataset(dataset['test'])

    else:
        raise ValueError("Unsupported task name")

    return train_dataset, test_dataset

def partition_dataset(dataset, token, num_classes, num_clients=64, alpha=0.5, batch_size=32, task_name=None, model_name=None):
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array([item['label'] for item in dataset])
    elif isinstance(dataset, CustomDataset):
        labels = np.array([item['label'] for item in dataset.dataset])
    else:
        raise AttributeError("Dataset does not have 'targets' or 'labels' attribute")

    data_indices = np.arange(len(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = data_indices[labels == c]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_split = np.split(class_indices, proportions)

        for i in range(num_clients):
            client_indices[i].extend(client_split[i])

    client_data_loaders = []
    for indices in client_indices:
        indices = [int(idx) for idx in indices]
        client_data = Subset(dataset, indices)
        if task_name in ["AGNEWS", "YELP-F"]:
            client_data_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: collate_fn(x, token, model_name))
        else:
            client_data_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True, drop_last=True)
        client_data_loaders.append(client_data_loader)

    return client_data_loaders

def load_client_Data(task_name, token, model_name, Clients, alpha=0.5, batchsize=128):
    if task_name == "CIFAR-100":
        num_classes = 100
    elif task_name == "Tiny ImageNet":
        num_classes = 200
    elif task_name == "AGNEWS":
        num_classes = 4
    elif task_name == "YELP-F":
        num_classes = 5
    else:
        raise ValueError("Unsupported task name")

    TrainDataset, TestDataset = load_specific_dataset(task_name)

    if task_name in ["AGNEWS", "YELP-F"]:
        test_loader = DataLoader(TestDataset, batch_size=128, shuffle=False, collate_fn=lambda x: collate_fn(x, token, model_name))
    else:
        test_loader = DataLoader(TestDataset, batch_size=128, shuffle=False)

    client_train_loaders = partition_dataset(TrainDataset, token, num_classes, num_clients=Clients, alpha=alpha, batch_size=batchsize, task_name=task_name, model_name=model_name)
    client_test_loaders = partition_dataset(TestDataset, token, num_classes, num_clients=Clients, alpha=alpha, batch_size=batchsize, task_name=task_name, model_name=model_name)

    return test_loader, client_train_loaders, client_test_loaders


