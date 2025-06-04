import wandb
import time
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from scripts.preprocessing.data_loader import ThermalDataset
from scripts.models.R2AU_dynamic import R2AttU_Net, init_weights, EarlyStopping
from scripts.models.losses import *
from scripts.models.metrics import *
from scripts.models.metrics_multiclass import get_multiclass_metrics
from scripts.models.trainer import Trainer

# --- WANDB SETUP ---
wandb.init(
    project="multiclass_prova",
    name="prova1",
    config={
        "val_ratio": 0.1,
        "batch_size": 4,
        "init_type": "kaiming",
        "loss": "focal_tversky",
        "lr": 0.00085939,
        "n_epochs": 100,
        "t": 4,
        "base_filters": 16,
        "depth": 5,
        "alpha": 0.56814,
        "gamma": 1.52575,
        "accum_steps": 4,
        "task_type": "multiclass",  # "binary" o "multiclass"
    },
)
config = wandb.config

start_time = time.time()
torch.backends.cudnn.benchmark = True

# --- DATASET ---
images_dir = "/home/gabrielepassoni/Documents/BCXTT/data/images_numpy"
masks_dir = "/home/gabrielepassoni/Documents/BCXTT/data/mask_preprocessed"

dataset = ThermalDataset(
    images_dir=images_dir,
    masks_dir=masks_dir,
    height=256,
    width=384,
    multiclass=(config.task_type == "multiclass"),
)

val_size = int(len(dataset) * config.val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
)

# --- MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_ch = 3 if config.task_type == "multiclass" else 1

model = R2AttU_Net(
    img_ch=1,
    output_ch=output_ch,
    t=config.t,
    base_filters=config.base_filters,
    depth=config.depth,
).to(device, non_blocking=True)
init_weights(model, init_type=config.init_type)

# --- LOSS SETUP ---
if config.task_type == "binary":
    criterion = FocalTverskyLoss(
        alpha=config.alpha, beta=1 - config.alpha, gamma=config.gamma
    )
else:
    criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=config.lr)
early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)

# --- TRAINING ---
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device,
    early_stopping=early_stopping,
    task_type=config.task_type,
)

best_model, best_metrics_train = trainer.run()

print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
