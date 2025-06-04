import wandb
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from scripts.preprocessing.data_loader import ThermalDataset
from scripts.models.R2AU_dynamic import R2AttU_Net, init_weights, EarlyStopping
from scripts.models.losses import *
from scripts.models.metrics import *
from scripts.models.trainer import Trainer


def log_error_to_file(msg):
    with open("errors_hyper.txt", "a") as f:
        f.write(msg + "\n")


def main():
    wandb.init()
    config = wandb.config
    config.val_ratio = 0.1
    config.init_type = "kaiming"
    config.n_epochs = 100
    config.accum_steps = 4
    config.batch_size = 4

    torch.backends.cudnn.benchmark = True

    # --- DATASET ---
    images_dir = "/home/gabrielepassoni/Documents/BCXTT/data/images_numpy"
    masks_dir = "/home/gabrielepassoni/Documents/BCXTT/data/mask_preprocessed"

    dataset = ThermalDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        height=256,
        width=384,
        multiclass=True,
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
    output_ch = 3
    model = R2AttU_Net(
        img_ch=1,
        output_ch=output_ch,
        t=config.t,
        base_filters=config.base_filters,
        depth=config.depth,
    ).to(device, non_blocking=True)
    init_weights(model, init_type=config.init_type)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)

    # --- TRAIN ---
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        early_stopping=early_stopping,
        task_type="multiclass",
    )
    trainer.run()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ Skipping run due to CUDA OOM")
            torch.cuda.empty_cache()  # libera memoria non usata
            torch.cuda.ipc_collect()

            cfg = wandb.config
            params = {
                "batch_size": getattr(cfg, "batch_size", "NA"),
                "base_filters": getattr(cfg, "base_filters", "NA"),
                "depth": getattr(cfg, "depth", "NA"),
                "learning_rate": getattr(cfg, "learning_rate", "NA"),
                "t": getattr(cfg, "t", "NA"),
                "gamma": getattr(cfg, "gamma", "NA"),
                "alpha": getattr(cfg, "alpha", "NA"),
            }
            log_msg = f"[OOM] Params: " + ", ".join(
                f"{k}={v}" for k, v in params.items()
            )
            log_error_to_file(log_msg)
        else:
            log_error_to_file(f"[RUNTIME ERROR] {str(e)}")
            raise e
    except Exception as e:
        log_error_to_file(f"[GENERAL ERROR] {str(e)}")
        raise e
    finally:
        wandb.finish(exit_code=1)
