import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics import Accuracy
from torch import nn
import pytorch_lightning as pl


learning_rate = 0.01
batch_size = 64
num_epochs = 15

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(224, padding=4),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((224, 224), antialias=True),
    ]
)


class PreTrainedClassificationModel(pl.LightningModule):

    def __init__(self, num_classes):
        super(PreTrainedClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.train_acc_plot = []
        self.train_loss_plot = []
        self.val_acc_plot = []
        self.val_loss_plot = []
        self.train_loss_outputs = []
        self.train_acc_outputs = []
        self.val_loss_outputs = []
        self.val_acc_outputs = []

        self.train_acc = Accuracy(task="multiclass", num_classes=4)
        self.val_acc = Accuracy(task="multiclass", num_classes=4)

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)
        # self.backbone = efficientnet_b0(pretrained=True)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        out = self.backbone(x)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        labels = labels.view(-1)
        loss = nn.functional.cross_entropy(outputs, labels)
        acc = self.train_acc(outputs, labels)
        self.train_acc_outputs.append(acc)
        self.train_loss_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        labels = labels.view(-1)
        loss = nn.functional.cross_entropy(outputs, labels)
        acc = self.val_acc(outputs, labels)
        self.val_acc_outputs.append(acc)
        self.val_loss_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        avg_train_acc = torch.mean(torch.tensor(self.train_acc_outputs))
        avg_train_loss = torch.mean(torch.tensor(self.train_loss_outputs))
        self.log("train_acc", avg_train_acc, prog_bar=True)
        self.log("train_loss", avg_train_loss, prog_bar=True)

        self.train_acc_plot.append(avg_train_acc)
        self.train_loss_plot.append(avg_train_loss)

        print(f"train_acc: {avg_train_acc:.2f} / train_loss: {avg_train_loss:.2f}")
        print(50 * "-")

        self.train_acc_outputs.clear()
        self.train_loss_outputs.clear()

    def on_validation_epoch_end(self):
        avg_val_acc = torch.mean(torch.tensor(self.val_acc_outputs))
        avg_val_loss = torch.mean(torch.tensor(self.val_loss_outputs))
        self.log("val_acc", avg_val_acc, prog_bar=True)
        self.log("val_loss", avg_val_loss, prog_bar=True)

        self.val_acc_plot.append(avg_val_acc)
        self.val_loss_plot.append(avg_val_loss)

        print(f"Epoch: {self.current_epoch+1}/{num_epochs}")
        print(f"val_acc: {avg_val_acc:.2f} / val_loss: {avg_val_loss:.2f}")

        self.val_acc_outputs.clear()
        self.val_loss_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(
            self.parameters(), lr=learning_rate, lr_decay=0.9, weight_decay=0.001
        )
        return optimizer

    def train_dataloader(self):
        train_data = ImageFolder(
            "/kaggle/input/chest-ct-images/Data/train", transform=train_transform
        )
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_data = ImageFolder(
            "/kaggle/input/chest-ct-images/Data/valid", transform=val_transform
        )
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        return val_dataloader
