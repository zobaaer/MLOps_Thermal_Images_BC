import torch
import torchvision
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from .metrics import get_metrics
from .metrics_multiclass import get_multiclass_metrics
import time
import shutil
from pathlib import Path
from datetime import datetime
import json


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        config,
        device,
        early_stopping=None,
        task_type="binary",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.task_type = task_type
        self.early_stopping = early_stopping
        self.scaler = GradScaler()  # For AMP
        self.best_metrics = {
            "f1": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
        self.best_last_dir = Path("best_last_model")
        self.best_last_dir.mkdir(exist_ok=True)
        self.best_overall_dir = Path("best_model_saved")
        self.best_overall_dir.mkdir(exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_model_path = self.best_last_dir / f"run_{self.run_id}_best_model.pth"
        self.best_overall_metric = self.load_best_model_metric()

    def best_metrics_tracker(self, current_dict):
        if current_dict["f1"] > self.best_metrics["f1"]:
            self.best_metrics = current_dict
            torch.save(self.model.state_dict(), self.best_model_path)

    def load_best_model_metric(self):
        metric_file = self.best_overall_dir / "metric.txt"
        if metric_file.exists():
            try:
                with open(metric_file, "r") as f:
                    return float(f.read().strip())
            except:
                pass
        return 0.0

    def update_best_overall(self):
        """Controlla metric.txt, crea o aggiorna la metrica e copia il .pth se la run corrente supera il best global."""
        metric_file = self.best_overall_dir / "metric.txt"
        best_overall = 0.0
        if metric_file.exists():
            try:
                with open(metric_file, "r") as f:
                    best_overall = float(f.read().strip())
            except (ValueError, IOError):
                print("Error reading the best overall metric file. Defaulting to 0.0.")
                best_overall = 0.0

        current = self.best_metrics["f1"]
        if current > best_overall:
            # 1. Salva i pesi
            torch.save(
                self.model.state_dict(), self.best_overall_dir / "best_model.pth"
            )

            # 2. Salva la config in JSON
            import json

            with open(self.best_overall_dir / "config.json", "w") as f:
                json.dump(dict(self.config), f, indent=4)

            # 3. Salva anche la metrica come prima
            with open(self.best_overall_dir / "metric.txt", "w") as f:
                f.write(f"{current:.6f}")

            with open(metric_file, "w") as f:
                f.write(f"{current:.6f}")

    def run(self):
        self.optimizer.zero_grad()

        for epoch in range(self.config.n_epochs):
            epoch_duration_start = time.time()
            self.model.train()
            train_loss = 0.0

            for step, (images, masks) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with autocast():  # Mixed precision context
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks) / self.config.accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.config.accum_steps == 0 or (
                    step + 1 == len(self.train_loader)
                ):
                    # Gradient clipping (optional but recommended with AMP)
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(self.train_loader.dataset)

            val_loss, metrics = self.validate()
            self.best_metrics_tracker(metrics)

            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    **metrics,
                    "epoch": epoch,
                    "epoch_duration": (time.time() - epoch_duration_start) / 60,
                }
            )

            if epoch % 2 == 0:
                with torch.no_grad():
                    mask = masks[0].cpu()
                    if self.task_type == "binary":
                        pred = torch.sigmoid(outputs[0].detach()).cpu()
                    else:
                        pred = torch.argmax(outputs[0].detach(), dim=0, keepdim=True).float().cpu()
                        if mask.ndim == 2:  # da [H, W] → [1, H, W]
                            mask = mask.unsqueeze(0)

                    comparison = torch.stack([mask, pred], dim=0)
                    grid = torchvision.utils.make_grid(
                        comparison, nrow=2, normalize=False
                    )
                    wandb.log(
                        {
                            "Val Comparison [Ground Truth| Pred]": [
                                wandb.Image(grid, caption=f"Epoch {epoch}")
                            ]
                        }
                    )

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered. Training stopped")
                    break

        wandb.finish()

        self.update_best_overall()

        best_run_path = self.best_last_dir / f"run_{self.run_id}_best_model.pth"
        self.model.load_state_dict(torch.load(best_run_path))
        return self.model, self.best_metrics

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(masks.cpu())

        val_loss /= len(self.val_loader.dataset)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if self.task_type == "binary":
            metrics = get_metrics(all_preds, all_targets)
        else:
            metrics = get_multiclass_metrics(all_preds, all_targets)

        return val_loss, metrics
