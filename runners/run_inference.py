import torch
from ogb.graphproppred import Evaluator
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from datasets.datasets import get_data
from models import models
from reproducibility.utils import set_seeds
from train import evaluate, train


def run_inference(args, device):

    set_seeds(args.seed)

    # Get Data
    train_data, val_data, test_data, stats = get_data(args.dataset, perslay_feats=args.gnn=='linear')
    if args.gnn == 'linear':
        args.n_graph_features = train_data.graph_features.shape[1]

    args.num_node_features = stats["num_features"]
    args.num_classes = stats["num_classes"]

    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    if args.dataset == "ZINC":
        loss_fn = torch.nn.L1Loss(reduction='mean')

    evaluator = None
    if args.dataset == "ogbg-molhiv":
        evaluator = Evaluator(args.dataset)


    model = models.get_model(args).to(device)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

   
    start_time = time.time()
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, evaluator) 
    end_time = time.time()
    
    return end_time

