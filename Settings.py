import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import random as rd
import copy as cp

import gc
import time
import math
from collections import OrderedDict, defaultdict
import clip
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, DistilBertTokenizer, DistilBertModel
from transformers import CLIPProcessor, CLIPModel
from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import models
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

seed = 1234 # 1, 12, 123, 1234
np.random.seed(seed)
rd.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True








