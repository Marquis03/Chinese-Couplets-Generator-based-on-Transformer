import os
import gc
import time
import math
import random
import joblib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sns.set_theme(
    style="darkgrid", font_scale=1.2, font="SimHei", rc={"axes.unicode_minus": False}
)

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR


from config import Config
from model import CoupletsTransformer
from dataset import load_data, Vocab, CoupletsDataset
from utils import EarlyStopping


def train_model(
    config, model, train_loader, val_loader, optimizer, criterion, scheduler
):
    model = model.to(config.device)
    best_loss = float("inf")
    history = []
    model_path = os.path.join(config.model_save_dir, f"{model.name}_best.pth")
    if config.early_stop:
        early_stopping = EarlyStopping(patience=config.patience, delta=config.delta)
    for epoch in tqdm(range(1, config.epochs + 1), desc=f"All"):
        train_loss = train_one_epoch(
            config, model, train_loader, optimizer, criterion, scheduler
        )
        val_loss = evaluate(config, model, val_loader, criterion)

        perplexity = math.exp(val_loss)
        history.append((epoch, train_loss, val_loss))
        msg = f"Epoch {epoch}/{config.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Perplexity: {perplexity:.4f}"
        logger.info(msg)
        if val_loss < best_loss:
            logger.info(
                f"Val loss decrease from {best_loss:>10.6f} to {val_loss:>10.6f}"
            )
            torch.save(model.state_dict(), model_path)
            best_loss = val_loss
        if config.early_stop:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    logger.info(f"Save best model with val loss {best_loss:.6f} to {model_path}")

    model_path = os.path.join(config.model_save_dir, f"{model.name}_last.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Save last model with val loss {val_loss:.6f} to {model_path}")

    history = pd.DataFrame(
        history, columns=["Epoch", "Train Loss", "Val Loss"]
    ).set_index("Epoch")
    history.plot(
        subplots=True, layout=(1, 2), sharey="row", figsize=(14, 6), marker="o", lw=2
    )
    history_path = os.path.join(config.img_save_dir, "history.png")
    plt.savefig(history_path, dpi=300)
    logger.info(f"Save history to {history_path}")


def train_one_epoch(config, model, train_loader, optimizer, criterion, scheduler):
    model.train()
    train_loss = 0
    for src, tgt in tqdm(train_loader, desc=f"Epoch", leave=False):
        src, tgt = src.to(config.device), tgt.to(config.device)
        output = model(src, tgt[:, :-1], config.PAD_IDX)
        output = output.contiguous().view(-1, output.size(-1))
        tgt = tgt[:, 1:].contiguous().view(-1)
        loss = criterion(output, tgt)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return train_loss / len(train_loader)


def evaluate(config, model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc=f"Val", leave=False):
            src, tgt = src.to(config.device), tgt.to(config.device)
            output = model(src, tgt[:, :-1], config.PAD_IDX)
            output = output.contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def test_model(model, data, vocab):
    model.eval()
    for src_text, tgt_text in data:
        src_text, tgt_text = "".join(src_text), "".join(tgt_text)
        out_text = model.generate(src_text, vocab)
        logger.info(f"\nInput: {src_text}\nTarget: {tgt_text}\nOutput: {out_text}")


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    config = Config()

    # Set random seed
    seed_everything(config.seed)
    logger.info(f"Set random seed to {config.seed}")

    # Set cuDNN
    if config.cuDNN:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Load data
    data = load_data([config.in_path, config.out_path])
    if config.debug:
        data = data[:1000]
    logger.info(f"Load {len(data)} couplets")

    # Build vocab
    vocab = Vocab(data)
    vocab_size = len(vocab)
    logger.info(f"Build vocab with {vocab_size} tokens")
    vocab_path = os.path.join(config.model_save_dir, "vocab.pkl")
    joblib.dump(vocab, vocab_path)
    logger.info(f"Save vocab to {vocab_path}")

    # Build dataset
    data_train, data_val = train_test_split(
        data, test_size=config.val_ratio, random_state=config.seed, shuffle=True
    )
    train_dataset = CoupletsDataset(data_train, vocab)
    val_dataset = CoupletsDataset(data_val, vocab)

    config.PAD_IDX = train_dataset.PAD_IDX

    logger.info(f"Build train dataset with {len(train_dataset)} samples")
    logger.info(f"Build val dataset with {len(val_dataset)} samples")

    # Build dataloader
    train_loader = train_dataset.get_loader(
        config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = val_dataset.get_loader(
        config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    logger.info(f"Build train dataloader with {len(train_loader)} batches")
    logger.info(f"Build val dataloader with {len(val_loader)} batches")

    # Build model
    model = CoupletsTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        nhead=config.num_head,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )
    logger.info(f"Build model with {model.name}")

    # Build optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )

    # Build criterion
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX, reduction="mean")

    # Build scheduler
    lr_max, lr_min = config.lr_max, config.lr_min
    T_max = config.epochs * len(train_loader)
    warm_up_iter = int(T_max * config.warmup_ratio)

    def WarmupExponentialLR(cur_iter):
        gamma = math.exp(math.log(lr_min / lr_max) / (T_max - warm_up_iter))
        if cur_iter < warm_up_iter:
            return (lr_max - lr_min) * (cur_iter / warm_up_iter) + lr_min
        else:
            return lr_max * gamma ** (cur_iter - warm_up_iter)

    scheduler = LambdaLR(optimizer, lr_lambda=WarmupExponentialLR)

    df_lr = pd.DataFrame(
        [WarmupExponentialLR(i) for i in range(T_max)],
        columns=["Learning Rate"],
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_lr, linewidth=2)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    lr_img_path = os.path.join(config.img_save_dir, "lr_schedule.png")
    plt.savefig(lr_img_path, dpi=300)
    logger.info(f"Save learning rate schedule to {lr_img_path}")

    # Garbage collect
    gc.collect()
    torch.cuda.empty_cache()

    # Train model
    train_model(
        config, model, train_loader, val_loader, optimizer, criterion, scheduler
    )

    # Test model
    test_model(model, data_val[:10], vocab)


if __name__ == "__main__":
    main()
