# train.py
"""
Train script for combined MobileNetV2 + ResNet50 classifier.

This script will:
  - Load train/validation image folders via get_dataloaders()
  - Instantiate CombinedModel (frozen MobileNetV2 & ResNet50 heads + trainable classifier)
  - Train for a specified number of epochs, computing per-epoch:
      * loss, accuracy, precision, recall, F1-score, average class accuracy
  - Save metrics into `train_stats.txt` and `validation_stats.txt`
  - Plot metric curves after each epoch and save at 900 dpi PNG
  - Every 2 epochs, plot & save confusion matrices under `confusion_mat/`
  - Save a model checkpoint at the end of each epoch (`model_epoch_{epoch}.pth`)
  - Display batch‐level progress with tqdm bars

Usage:
    python train.py \
        --data_dir /path/to/data \
        --output_dir ./outputs \
        --epochs 20 \
        --batch_size 32
"""

import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import CombinedModel
from dataloader import get_dataloaders
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix
)


def save_stats(stats: dict, filepath: str):
    """
    Append a line of comma‐separated stats to the given file.
    """
    line = ','.join(f'{v:.4f}' if isinstance(v, float) else str(v)
                    for v in stats.values())
    with open(filepath, 'a') as f:
        f.write(line + '\n')


def plot_metric(train_vals, val_vals, metric_name: str, output_dir: str):
    """
    Plot train vs. validation curves for a single metric and save as PNG.
    """
    plt.figure()
    plt.plot(range(1, len(train_vals)+1), train_vals, label='train')
    plt.plot(range(1, len(val_vals)+1),   val_vals,   label='val')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric_name}.png'), dpi=900)
    plt.close()


def plot_confusion(cm, classes, title: str, filepath: str):
    """
    Render and save a confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > thresh else "black"
        plt.text(j, i, cm[i, j], ha="center", va="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=900)
    plt.close()


def run_epoch(model, loader, criterion, optimizer, device, mode: str):
    """
    Run one epoch of training or validation.
    Returns:
      - metrics dict with keys:
          epoch, loss, accuracy, precision, recall, f1_score, avg_class_accuracy
      - confusion matrix (np.ndarray)
    """
    is_train = (optimizer is not None)
    model.train() if is_train else model.eval()

    all_preds = []
    all_labels = []
    running_loss = 0.0

    desc = f"{mode.capitalize()} ({'train' if is_train else 'val'})"
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels.cpu())

    # aggregate
    total_samples = len(loader.dataset)
    epoch_loss = running_loss / total_samples

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    avg_cls_acc = class_acc.mean()

    metrics = {
        'epoch':    None,           # to be set by caller
        'loss':     epoch_loss,
        'accuracy': acc,
        'precision':prec,
        'recall':   rec,
        'f1_score': f1,
        'avg_class_accuracy': avg_cls_acc
    }
    return metrics, cm


def main(args):
    # device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    cm_dir = os.path.join(args.output_dir, 'confusion_mat')
    os.makedirs(cm_dir, exist_ok=True)

    # load data
    train_loader, val_loader, classes = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, args.img_size
    )

    # build model + optimizer
    model = CombinedModel(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # init stats files
    headers = ['epoch','loss','accuracy','precision','recall','f1_score','avg_class_accuracy']
    for fname in ('train_stats.txt', 'validation_stats.txt'):
        with open(os.path.join(args.output_dir, fname), 'w') as f:
            f.write(','.join(headers) + '\n')

    # history buffers for plotting
    history = {
        'train': {h: [] for h in headers if h != 'epoch'},
        'val':   {h: [] for h in headers if h != 'epoch'}
    }

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_metrics, train_cm = run_epoch(
            model, train_loader, criterion, optimizer, device, mode='train'
        )
        val_metrics, val_cm = run_epoch(
            model, val_loader, criterion, None, device, mode='val'
        )

        # set epoch number
        train_metrics['epoch'] = val_metrics['epoch'] = epoch

        # save stats
        save_stats(train_metrics, os.path.join(args.output_dir, 'train_stats.txt'))
        save_stats(val_metrics,   os.path.join(args.output_dir, 'validation_stats.txt'))

        # update history for plotting
        for k in history['train']:
            history['train'][k].append(train_metrics[k])
        for k in history['val']:
            history['val'][k].append(val_metrics[k])

        # plot metrics curves
        for metric in history['train']:
            plot_metric(
                history['train'][metric],
                history['val'][metric],
                metric,
                args.output_dir
            )

        # save confusion matrices every 2 epochs
        if epoch % 2 == 0:
            plot_confusion(
                train_cm, classes,
                title=f'Train CM Epoch {epoch}',
                filepath=os.path.join(cm_dir, f'train_epoch_{epoch}.png')
            )
            plot_confusion(
                val_cm, classes,
                title=f'Val CM Epoch {epoch}',
                filepath=os.path.join(cm_dir, f'val_epoch_{epoch}.png')
            )

        # save model checkpoint
        ckpt_path = os.path.join(args.output_dir, f'model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss']
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # epoch summary
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f}")

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train combined MobileNetV2 + ResNet50 classification network."
    )
    parser.add_argument('--data_dir',   required=True,
                        help="Root directory with 'train/' and 'val/' subfolders.")
    parser.add_argument('--output_dir', default='./outputs',
                        help="Directory to save stats, plots, confusion matrices, and checkpoints.")
    parser.add_argument('--epochs',     type=int,   default=10,
                        help="Number of epochs to train.")
    parser.add_argument('--batch_size', type=int,   default=32,
                        help="Batch size for train & val loaders.")
    parser.add_argument('--lr',         type=float, default=1e-4,
                        help="Learning rate for optimizer.")
    parser.add_argument('--num_workers',type=int,   default=4,
                        help="Number of parallel data-loading workers.")
    parser.add_argument('--img_size',   type=int,   default=224,
                        help="Height & width to resize images to.")
    args = parser.parse_args()
    main(args)
