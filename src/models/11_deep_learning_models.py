# -*- coding: utf-8 -*-
"""
第三项：深度学习模型构建与优化
功能：
1. 从用户行为明细构建“用户-商品未来7天是否购买”的监督学习样本；
2. 构建用户历史行为序列，并完成类别特征Embedding；
3. 实现LSTM、GRU、DIN三种模型；
4. 支持Embedding维度、隐藏层单元数、Dropout、多头注意力等参数配置；
5. 使用学习率衰减、Early Stopping防止过拟合；
6. 输出AUC、准确率、召回率、F1，并保存模型、预测结果和对比报告；
7. 可选使用Optuna自动调参；
8. 可选读取传统模型预测结果，与深度学习模型进行性能对比。

默认输入字段：
user_id,item_id,item_category,behavior_type,time
其中behavior_type=4表示购买。
如你的字段名不同，请在CONFIG中修改。
"""

import os
import gc
import json
import math
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass,asdict
from typing import Dict,List,Tuple,Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset,DataLoader
except ImportError as e:
    raise ImportError("请先安装PyTorch：pip install torch") from e

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    raise ImportError("请先安装scikit-learn：pip install scikit-learn") from e


# ============================================================
# 1. 全局配置
# ============================================================
@dataclass
class Config:
    data_path:str="../../data/processed/user_behavior_cleaned.parquet"
    output_dir:str="results/deep_learning"
    traditional_result_path:str=""

    user_col:str="user_id"
    item_col:str="item_id"
    category_col:str="item_category"
    behavior_col:str="behavior_type"
    time_col:str="time"
    buy_behavior:int=4

    history_days:int=7
    label_days:int=3
    max_seq_len:int=50
    negative_ratio:int=3
    max_rows:int=0

    train_ratio:float=0.70
    val_ratio:float=0.20
    test_ratio:float=0.10

    model_name:str="all"       # lstm/gru/din/all
    embed_dim:int=32
    hidden_dim:int=64
    num_layers:int=1
    dropout:float=0.30
    num_heads:int=4

    batch_size:int=512
    epochs:int=20
    learning_rate:float=1e-3
    weight_decay:float=1e-5
    patience:int=4
    num_workers:int=0

    use_class_weight:bool=True
    use_optuna:bool=False
    optuna_trials:int=10

    seed:int=42
    device:str="cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. 基础工具
# ============================================================
def set_seed(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def ensure_dir(path:str)->None:
    Path(path).mkdir(parents=True,exist_ok=True)


def load_data(cfg:Config)->pd.DataFrame:
    path=Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            "请修改CONFIG中的data_path，或运行时使用：--data_path 你的文件路径"
        )

    usecols=[cfg.user_col,cfg.item_col,cfg.category_col,cfg.behavior_col,cfg.time_col]
    print(f"[1/8] 正在读取数据：{path}")

    if path.suffix.lower()==".parquet":
        df=pd.read_parquet(path,columns=usecols)
    elif path.suffix.lower()==".csv":
        df=pd.read_csv(path,usecols=usecols)
    else:
        raise ValueError("仅支持.csv或.parquet文件")

    if cfg.max_rows>0 and len(df)>cfg.max_rows:
        df=df.sample(cfg.max_rows,random_state=cfg.seed)

    df=df.dropna(subset=usecols).copy()
    df[cfg.time_col]=pd.to_datetime(df[cfg.time_col],errors="coerce")
    df=df.dropna(subset=[cfg.time_col])
    df[cfg.behavior_col]=pd.to_numeric(df[cfg.behavior_col],errors="coerce")
    df=df.dropna(subset=[cfg.behavior_col])
    df[cfg.behavior_col]=df[cfg.behavior_col].astype(np.int8)
    df=df.sort_values(cfg.time_col).reset_index(drop=True)

    print(f"数据读取完成：{len(df):,}行，时间范围："
          f"{df[cfg.time_col].min()} 至 {df[cfg.time_col].max()}")
    return df


# ============================================================
# 3. 构建未来7天购买标签与历史序列
# ============================================================
def choose_cutoff_times(df:pd.DataFrame,cfg:Config)->Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp]:
    min_time=df[cfg.time_col].min().normalize()
    max_time=df[cfg.time_col].max().normalize()
    usable_start=min_time+pd.Timedelta(days=cfg.history_days)
    usable_end=max_time-pd.Timedelta(days=cfg.label_days)

    if usable_end<=usable_start:
        raise ValueError(
            f"数据时间跨度不足。至少需要history_days({cfg.history_days})+"
            f"label_days({cfg.label_days})天。"
        )

    total_days=max((usable_end-usable_start).days,3)
    train_cut=usable_start+pd.Timedelta(days=max(1,int(total_days*cfg.train_ratio)))
    val_cut=usable_start+pd.Timedelta(days=max(2,int(total_days*(cfg.train_ratio+cfg.val_ratio))))
    val_cut=min(val_cut,usable_end-pd.Timedelta(days=1))

    return train_cut,val_cut,usable_end


def encode_ids(df:pd.DataFrame,cfg:Config):
    print("[2/8] 正在编码用户、商品、类别和行为ID")
    encoders={}
    for col in [cfg.user_col,cfg.item_col,cfg.category_col,cfg.behavior_col]:
        encoder=LabelEncoder()
        df[col+"_idx"]=encoder.fit_transform(df[col].astype(str))+1
        encoders[col]=encoder
    return df,encoders


def build_one_split(
    df:pd.DataFrame,
    cutoff:pd.Timestamp,
    split_name:str,
    cfg:Config
)->pd.DataFrame:
    history_start=cutoff-pd.Timedelta(days=cfg.history_days)
    label_end=cutoff+pd.Timedelta(days=cfg.label_days)

    history=df[(df[cfg.time_col]>=history_start)&(df[cfg.time_col]<cutoff)].copy()
    future=df[(df[cfg.time_col]>=cutoff)&(df[cfg.time_col]<label_end)].copy()

    if history.empty:
        raise ValueError(f"{split_name}历史窗口没有数据：{history_start}~{cutoff}")

    # 候选样本：历史窗口中发生过交互的用户-商品对
    candidates=history[
        [cfg.user_col+"_idx",cfg.item_col+"_idx",cfg.category_col+"_idx"]
    ].drop_duplicates()

    # 未来7天购买标签
    buyers=future[future[cfg.behavior_col]==cfg.buy_behavior][
        [cfg.user_col+"_idx",cfg.item_col+"_idx"]
    ].drop_duplicates()
    buyers["label"]=1

    samples=candidates.merge(
        buyers,on=[cfg.user_col+"_idx",cfg.item_col+"_idx"],how="left"
    )
    samples["label"]=samples["label"].fillna(0).astype(np.int8)

    # 负样本欠采样，缓解类别不平衡并减少训练时间
    positives=samples[samples["label"]==1]
    negatives=samples[samples["label"]==0]

    if len(positives)>0 and cfg.negative_ratio>0:
        n_neg=min(len(negatives),len(positives)*cfg.negative_ratio)
        negatives=negatives.sample(n=n_neg,random_state=cfg.seed)
        samples=pd.concat([positives,negatives],ignore_index=True)
    else:
        samples=samples.copy()

    samples=samples.sample(frac=1,random_state=cfg.seed).reset_index(drop=True)
    samples["cutoff_time"]=cutoff
    samples["split"]=split_name

    # 用户历史行为序列：按时间顺序保留最近max_seq_len条
    history=history.sort_values(cfg.time_col)
    user_sequences={}
    for uid,g in history.groupby(cfg.user_col+"_idx",sort=False):
        tail=g.tail(cfg.max_seq_len)
        user_sequences[int(uid)]={
            "item_seq":tail[cfg.item_col+"_idx"].astype(np.int64).tolist(),
            "cate_seq":tail[cfg.category_col+"_idx"].astype(np.int64).tolist(),
            "behavior_seq":tail[cfg.behavior_col+"_idx"].astype(np.int64).tolist()
        }

    def get_seq(uid:int,key:str)->List[int]:
        return user_sequences.get(int(uid),{}).get(key,[])

    samples["item_seq"]=samples[cfg.user_col+"_idx"].map(lambda x:get_seq(x,"item_seq"))
    samples["cate_seq"]=samples[cfg.user_col+"_idx"].map(lambda x:get_seq(x,"cate_seq"))
    samples["behavior_seq"]=samples[cfg.user_col+"_idx"].map(lambda x:get_seq(x,"behavior_seq"))

    pos_rate=samples["label"].mean() if len(samples)>0 else 0
    print(
        f"{split_name}: cutoff={cutoff.date()}, 样本={len(samples):,}, "
        f"正样本={samples['label'].sum():,}, 正样本率={pos_rate:.4f}"
    )
    return samples


def build_datasets(df:pd.DataFrame,cfg:Config)->pd.DataFrame:
    train_cut,val_cut,test_cut=choose_cutoff_times(df,cfg)
    print("[3/8] 正在按时间窗口构建训练集、验证集、测试集")
    print(f"训练截止点：{train_cut.date()}")
    print(f"验证截止点：{val_cut.date()}")
    print(f"测试截止点：{test_cut.date()}")

    train_df=build_one_split(df,train_cut,"train",cfg)
    val_df=build_one_split(df,val_cut,"val",cfg)
    test_df=build_one_split(df,test_cut,"test",cfg)

    all_samples=pd.concat([train_df,val_df,test_df],ignore_index=True)
    return all_samples


# ============================================================
# 4. PyTorch数据集
# ============================================================
class BehaviorDataset(Dataset):
    def __init__(self,df:pd.DataFrame,cfg:Config):
        self.df=df.reset_index(drop=True)
        self.cfg=cfg

    def __len__(self):
        return len(self.df)

    @staticmethod
    def pad_sequence(seq:List[int],max_len:int)->Tuple[np.ndarray,int]:
        seq=seq[-max_len:]
        length=len(seq)
        arr=np.zeros(max_len,dtype=np.int64)
        if length>0:
            arr[-length:]=np.asarray(seq,dtype=np.int64)
        return arr,length

    def __getitem__(self,idx:int):
        row=self.df.iloc[idx]
        item_seq,seq_len=self.pad_sequence(row["item_seq"],self.cfg.max_seq_len)
        cate_seq,_=self.pad_sequence(row["cate_seq"],self.cfg.max_seq_len)
        behavior_seq,_=self.pad_sequence(row["behavior_seq"],self.cfg.max_seq_len)

        return {
            "user_id":torch.tensor(int(row[self.cfg.user_col+"_idx"]),dtype=torch.long),
            "target_item":torch.tensor(int(row[self.cfg.item_col+"_idx"]),dtype=torch.long),
            "target_cate":torch.tensor(int(row[self.cfg.category_col+"_idx"]),dtype=torch.long),
            "item_seq":torch.tensor(item_seq,dtype=torch.long),
            "cate_seq":torch.tensor(cate_seq,dtype=torch.long),
            "behavior_seq":torch.tensor(behavior_seq,dtype=torch.long),
            "seq_len":torch.tensor(max(seq_len,1),dtype=torch.long),
            "label":torch.tensor(float(row["label"]),dtype=torch.float32)
        }


# ============================================================
# 5. 模型定义：LSTM、GRU、DIN
# ============================================================
class MultiHeadSequenceAttention(nn.Module):
    def __init__(self,input_dim:int,num_heads:int,dropout:float):
        super().__init__()
        if input_dim%num_heads!=0:
            valid=[h for h in range(1,num_heads+1) if input_dim%h==0]
            num_heads=max(valid) if valid else 1
        self.attention=nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm=nn.LayerNorm(input_dim)

    def forward(self,x,mask):
        attn_out,_=self.attention(
            x,x,x,
            key_padding_mask=mask,
            need_weights=False
        )
        return self.norm(x+attn_out)


class SequenceModel(nn.Module):
    def __init__(
        self,
        model_type:str,
        n_users:int,
        n_items:int,
        n_categories:int,
        n_behaviors:int,
        cfg:Config
    ):
        super().__init__()
        self.model_type=model_type
        self.cfg=cfg

        self.user_embedding=nn.Embedding(n_users+1,cfg.embed_dim,padding_idx=0)
        self.item_embedding=nn.Embedding(n_items+1,cfg.embed_dim,padding_idx=0)
        self.cate_embedding=nn.Embedding(n_categories+1,cfg.embed_dim,padding_idx=0)
        self.behavior_embedding=nn.Embedding(n_behaviors+1,cfg.embed_dim,padding_idx=0)

        seq_input_dim=cfg.embed_dim*3
        self.sequence_attention=MultiHeadSequenceAttention(
            seq_input_dim,cfg.num_heads,cfg.dropout
        )

        rnn_cls=nn.LSTM if model_type=="lstm" else nn.GRU
        self.rnn=rnn_cls(
            input_size=seq_input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers>1 else 0
        )

        final_dim=cfg.hidden_dim+cfg.embed_dim*3
        self.mlp=nn.Sequential(
            nn.Linear(final_dim,cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim,max(cfg.hidden_dim//2,16)),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(max(cfg.hidden_dim//2,16),1)
        )

    def forward(self,batch):
        item_emb=self.item_embedding(batch["item_seq"])
        cate_emb=self.cate_embedding(batch["cate_seq"])
        behavior_emb=self.behavior_embedding(batch["behavior_seq"])
        seq=torch.cat([item_emb,cate_emb,behavior_emb],dim=-1)

        padding_mask=batch["item_seq"].eq(0)
        seq=self.sequence_attention(seq,padding_mask)

        lengths=batch["seq_len"].cpu()
        packed=nn.utils.rnn.pack_padded_sequence(
            seq,lengths,batch_first=True,enforce_sorted=False
        )
        _,hidden=self.rnn(packed)

        if self.model_type=="lstm":
            hidden=hidden[0]
        seq_repr=hidden[-1]

        user_emb=self.user_embedding(batch["user_id"])
        target_item_emb=self.item_embedding(batch["target_item"])
        target_cate_emb=self.cate_embedding(batch["target_cate"])

        x=torch.cat([seq_repr,user_emb,target_item_emb,target_cate_emb],dim=1)
        return self.mlp(x).squeeze(1)


class DINAttention(nn.Module):
    def __init__(self,embed_dim:int,hidden_dim:int,dropout:float):
        super().__init__()
        input_dim=embed_dim*4
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,max(hidden_dim//2,16)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim//2,16),1)
        )

    def forward(self,history,target,mask):
        target_expand=target.unsqueeze(1).expand_as(history)
        interaction=torch.cat(
            [history,target_expand,history-target_expand,history*target_expand],
            dim=-1
        )
        scores=self.net(interaction).squeeze(-1)
        scores=scores.masked_fill(mask,-1e9)
        weights=torch.softmax(scores,dim=1)
        weighted_history=torch.sum(history*weights.unsqueeze(-1),dim=1)
        return weighted_history


class DINModel(nn.Module):
    def __init__(
        self,
        n_users:int,
        n_items:int,
        n_categories:int,
        n_behaviors:int,
        cfg:Config
    ):
        super().__init__()
        self.cfg=cfg

        self.user_embedding=nn.Embedding(n_users+1,cfg.embed_dim,padding_idx=0)
        self.item_embedding=nn.Embedding(n_items+1,cfg.embed_dim,padding_idx=0)
        self.cate_embedding=nn.Embedding(n_categories+1,cfg.embed_dim,padding_idx=0)
        self.behavior_embedding=nn.Embedding(n_behaviors+1,cfg.embed_dim,padding_idx=0)

        # DIN兴趣注意力：历史商品对目标商品的相关性
        self.din_attention=DINAttention(cfg.embed_dim,cfg.hidden_dim,cfg.dropout)

        # 多头注意力进一步提取历史序列上下文
        self.multi_head=MultiHeadSequenceAttention(
            cfg.embed_dim,cfg.num_heads,cfg.dropout
        )

        final_dim=cfg.embed_dim*5
        self.mlp=nn.Sequential(
            nn.Linear(final_dim,cfg.hidden_dim*2),
            nn.BatchNorm1d(cfg.hidden_dim*2),
            nn.PReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim*2,cfg.hidden_dim),
            nn.PReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim,1)
        )

    def forward(self,batch):
        mask=batch["item_seq"].eq(0)
        history_item=self.item_embedding(batch["item_seq"])
        target_item=self.item_embedding(batch["target_item"])

        context_history=self.multi_head(history_item,mask)
        interest=self.din_attention(context_history,target_item,mask)

        # 行为和类别序列做掩码平均池化
        valid=(~mask).float().unsqueeze(-1)
        denom=valid.sum(dim=1).clamp_min(1.0)
        cate_history=(self.cate_embedding(batch["cate_seq"])*valid).sum(dim=1)/denom
        behavior_history=(self.behavior_embedding(batch["behavior_seq"])*valid).sum(dim=1)/denom

        user_emb=self.user_embedding(batch["user_id"])
        target_cate=self.cate_embedding(batch["target_cate"])

        x=torch.cat(
            [interest,user_emb,target_item,target_cate,cate_history+behavior_history],
            dim=1
        )
        return self.mlp(x).squeeze(1)


# ============================================================
# 6. 训练、验证、Early Stopping、学习率衰减
# ============================================================
def move_batch(batch:Dict[str,torch.Tensor],device:str):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}


def safe_auc(y_true:np.ndarray,y_prob:np.ndarray)->float:
    return roc_auc_score(y_true,y_prob) if len(np.unique(y_true))>1 else float("nan")


def calculate_metrics(y_true:np.ndarray,y_prob:np.ndarray)->Dict[str,float]:
    y_pred=(y_prob>=0.5).astype(int)
    return {
        "auc":safe_auc(y_true,y_prob),
        "accuracy":accuracy_score(y_true,y_pred),
        "precision":precision_score(y_true,y_pred,zero_division=0),
        "recall":recall_score(y_true,y_pred,zero_division=0),
        "f1":f1_score(y_true,y_pred,zero_division=0)
    }


@torch.no_grad()
def evaluate(model,loader,criterion,device):
    model.eval()
    losses=[]
    labels=[]
    probs=[]

    for batch in loader:
        batch=move_batch(batch,device)
        logits=model(batch)
        loss=criterion(logits,batch["label"])
        losses.append(loss.item())
        labels.extend(batch["label"].cpu().numpy())
        probs.extend(torch.sigmoid(logits).cpu().numpy())

    labels=np.asarray(labels,dtype=np.int64)
    probs=np.asarray(probs,dtype=np.float32)
    metrics=calculate_metrics(labels,probs)
    metrics["loss"]=float(np.mean(losses)) if losses else float("nan")
    return metrics,labels,probs


def train_model(
    model,
    train_loader,
    val_loader,
    cfg:Config,
    model_name:str,
    pos_weight:Optional[float]=None
):
    device=cfg.device
    model=model.to(device)

    if pos_weight is not None:
        criterion=nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=device)
        )
    else:
        criterion=nn.BCEWithLogitsLoss()

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode="max",factor=0.5,patience=1,min_lr=1e-6
    )

    best_auc=-np.inf
    best_epoch=0
    wait=0
    history=[]
    best_path=Path(cfg.output_dir)/f"{model_name}_best.pt"

    print(f"\n开始训练{model_name.upper()}，设备：{device}")
    for epoch in range(1,cfg.epochs+1):
        model.train()
        train_losses=[]

        for batch in train_loader:
            batch=move_batch(batch,device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch)
            loss=criterion(logits,batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics,_,_=evaluate(model,val_loader,criterion,device)
        train_loss=float(np.mean(train_losses))
        current_auc=val_metrics["auc"]
        scheduler.step(current_auc if not math.isnan(current_auc) else 0.0)

        row={
            "epoch":epoch,
            "train_loss":train_loss,
            "val_loss":val_metrics["loss"],
            "val_auc":current_auc,
            "val_accuracy":val_metrics["accuracy"],
            "val_recall":val_metrics["recall"],
            "val_f1":val_metrics["f1"],
            "lr":optimizer.param_groups[0]["lr"]
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_auc={current_auc:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        improved=not math.isnan(current_auc) and current_auc>best_auc+1e-4
        if improved:
            best_auc=current_auc
            best_epoch=epoch
            wait=0
            torch.save(model.state_dict(),best_path)
        else:
            wait+=1
            if wait>=cfg.patience:
                print(f"Early Stopping：验证集AUC连续{cfg.patience}轮未提升")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path,map_location=device))

    pd.DataFrame(history).to_csv(
        Path(cfg.output_dir)/f"{model_name}_training_history.csv",
        index=False,encoding="utf-8-sig"
    )
    print(f"{model_name.upper()}最佳轮次：{best_epoch}，最佳验证AUC：{best_auc:.4f}")
    return model


# ============================================================
# 7. Optuna自动调参
# ============================================================
def run_optuna(
    model_name:str,
    sizes:Dict[str,int],
    train_loader,
    val_loader,
    base_cfg:Config,
    pos_weight:Optional[float]
)->Dict:
    try:
        import optuna
    except ImportError as e:
        raise ImportError("启用Optuna前请安装：pip install optuna") from e

    def objective(trial):
        cfg=Config(**asdict(base_cfg))
        cfg.embed_dim=trial.suggest_categorical("embed_dim",[16,32,64])
        cfg.hidden_dim=trial.suggest_categorical("hidden_dim",[32,64,128])
        cfg.dropout=trial.suggest_float("dropout",0.1,0.5)
        cfg.learning_rate=trial.suggest_float("learning_rate",1e-4,3e-3,log=True)
        cfg.num_heads=trial.suggest_categorical("num_heads",[1,2,4])
        cfg.epochs=min(base_cfg.epochs,8)
        cfg.patience=2

        model=create_model(model_name,sizes,cfg)
        model=train_model(
            model,train_loader,val_loader,cfg,
            f"{model_name}_trial_{trial.number}",pos_weight
        )

        criterion=nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight,device=cfg.device)
        ) if pos_weight is not None else nn.BCEWithLogitsLoss()
        metrics,_,_=evaluate(model,val_loader,criterion,cfg.device)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics["auc"] if not math.isnan(metrics["auc"]) else 0.0

    study=optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=base_cfg.optuna_trials)
    print("Optuna最佳参数：",study.best_params)
    return study.best_params


# ============================================================
# 8. 模型创建、预测保存与对比
# ============================================================
def create_model(model_name:str,sizes:Dict[str,int],cfg:Config):
    if model_name in ("lstm","gru"):
        return SequenceModel(
            model_name,
            sizes["n_users"],
            sizes["n_items"],
            sizes["n_categories"],
            sizes["n_behaviors"],
            cfg
        )
    if model_name=="din":
        return DINModel(
            sizes["n_users"],
            sizes["n_items"],
            sizes["n_categories"],
            sizes["n_behaviors"],
            cfg
        )
    raise ValueError(f"不支持的模型：{model_name}")


def save_predictions(
    test_df:pd.DataFrame,
    labels:np.ndarray,
    probs:np.ndarray,
    model_name:str,
    cfg:Config
):
    out=test_df[
        [cfg.user_col+"_idx",cfg.item_col+"_idx",cfg.category_col+"_idx","cutoff_time"]
    ].copy()
    out["label"]=labels
    out["probability"]=probs
    out["prediction"]=(probs>=0.5).astype(int)
    out.to_csv(
        Path(cfg.output_dir)/f"{model_name}_test_predictions.csv",
        index=False,encoding="utf-8-sig"
    )


def compare_with_traditional(result_df:pd.DataFrame,cfg:Config)->pd.DataFrame:
    path=Path(cfg.traditional_result_path) if cfg.traditional_result_path else None
    if path and path.exists():
        old=pd.read_csv(path)
        rename_map={
            "model":"model",
            "auc":"auc",
            "accuracy":"accuracy",
            "precision":"precision",
            "recall":"recall",
            "f1":"f1"
        }
        old=old.rename(columns=rename_map)
        keep=[c for c in ["model","auc","accuracy","precision","recall","f1"] if c in old.columns]
        if "model" in keep:
            result_df=pd.concat([old[keep],result_df],ignore_index=True,sort=False)
    return result_df


# ============================================================
# 9. 主流程
# ============================================================
def main(cfg:Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    with open(Path(cfg.output_dir)/"config.json","w",encoding="utf-8") as f:
        json.dump(asdict(cfg),f,ensure_ascii=False,indent=2)

    df=load_data(cfg)
    df,encoders=encode_ids(df,cfg)

    sizes={
        "n_users":int(df[cfg.user_col+"_idx"].max()),
        "n_items":int(df[cfg.item_col+"_idx"].max()),
        "n_categories":int(df[cfg.category_col+"_idx"].max()),
        "n_behaviors":int(df[cfg.behavior_col+"_idx"].max())
    }

    samples=build_datasets(df,cfg)
    samples.to_pickle(Path(cfg.output_dir)/"deep_learning_samples.pkl")

    train_df=samples[samples["split"]=="train"].reset_index(drop=True)
    val_df=samples[samples["split"]=="val"].reset_index(drop=True)
    test_df=samples[samples["split"]=="test"].reset_index(drop=True)

    if train_df["label"].nunique()<2:
        raise ValueError("训练集只包含一个类别，无法训练。请扩大时间窗口或检查购买行为编码。")

    train_loader=DataLoader(
        BehaviorDataset(train_df,cfg),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader=DataLoader(
        BehaviorDataset(val_df,cfg),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader=DataLoader(
        BehaviorDataset(test_df,cfg),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    pos_weight=None
    if cfg.use_class_weight:
        y=train_df["label"].values
        classes=np.array([0,1])
        weights=compute_class_weight(class_weight="balanced",classes=classes,y=y)
        pos_weight=float(weights[1]/weights[0])
        print(f"[4/8] 使用类别权重，正类pos_weight={pos_weight:.4f}")

    model_names=["lstm","gru","din"] if cfg.model_name=="all" else [cfg.model_name]
    all_results=[]

    for model_name in model_names:
        local_cfg=Config(**asdict(cfg))

        if cfg.use_optuna:
            print(f"[5/8] 正在为{model_name.upper()}执行Optuna调参")
            best_params=run_optuna(
                model_name,sizes,train_loader,val_loader,local_cfg,pos_weight
            )
            for k,v in best_params.items():
                setattr(local_cfg,k,v)

        print(f"[6/8] 构建{model_name.upper()}模型")
        model=create_model(model_name,sizes,local_cfg)
        model=train_model(
            model,train_loader,val_loader,local_cfg,model_name,pos_weight
        )

        criterion=nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=local_cfg.device)
        ) if pos_weight is not None else nn.BCEWithLogitsLoss()

        print(f"[7/8] 正在测试{model_name.upper()}")
        test_metrics,labels,probs=evaluate(
            model,test_loader,criterion,local_cfg.device
        )
        test_metrics["model"]=model_name.upper()
        all_results.append(test_metrics)
        save_predictions(test_df,labels,probs,model_name,cfg)

        print(
            f"{model_name.upper()}测试结果："
            f"AUC={test_metrics['auc']:.4f}, "
            f"Accuracy={test_metrics['accuracy']:.4f}, "
            f"Recall={test_metrics['recall']:.4f}, "
            f"F1={test_metrics['f1']:.4f}"
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[8/8] 保存模型性能对比结果")
    result_df=pd.DataFrame(all_results)[
        ["model","auc","accuracy","precision","recall","f1","loss"]
    ]
    result_df=compare_with_traditional(result_df,cfg)
    result_df.to_csv(
        Path(cfg.output_dir)/"deep_learning_model_comparison.csv",
        index=False,encoding="utf-8-sig"
    )

    print("\n模型性能对比：")
    print(result_df.to_string(index=False))
    print(f"\n全部结果已保存到：{Path(cfg.output_dir).resolve()}")


def parse_args():
    parser=argparse.ArgumentParser(description="第三项：LSTM/GRU/DIN深度学习模型")
    parser.add_argument("--data_path",type=str,default=Config.data_path)
    parser.add_argument("--output_dir",type=str,default=Config.output_dir)
    parser.add_argument("--model_name",type=str,default=Config.model_name,
                        choices=["lstm","gru","din","all"])
    parser.add_argument("--epochs",type=int,default=Config.epochs)
    parser.add_argument("--batch_size",type=int,default=Config.batch_size)
    parser.add_argument("--max_rows",type=int,default=Config.max_rows)
    parser.add_argument("--use_optuna",action="store_true")
    parser.add_argument("--optuna_trials",type=int,default=Config.optuna_trials)
    parser.add_argument("--traditional_result_path",type=str,default="")
    return parser.parse_args()


if __name__=="__main__":
    args=parse_args()
    config=Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        use_optuna=args.use_optuna,
        optuna_trials=args.optuna_trials,
        traditional_result_path=args.traditional_result_path
    )
    main(config)
