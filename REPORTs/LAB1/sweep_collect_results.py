"""
sweep_collect_results.py

用途：
1) 自动遍历多组超参数，分别训练 regression / classification 两种模式。
2) 训练完成后，直接读取保存的权重并在本进程计算指标（回归：MAE/R2；分类：F1/AUC）。
3) 仅生成 results.csv 汇总文件；每个 run 的标准输出写入 runs/<run_name>/stdout.log。
4) 生成若干简单图表（基于 Matplotlib），保存在 figures/ 目录。


使用方式（推荐在 LABs/LAB1 目录下）：
    python sweep_collect_results.py

若在仓库根目录运行，请指定训练脚本路径：
    python LABs/LAB1/sweep_collect_results.py --train-script LABs/LAB1/train.py

先决条件：

注意：
"""
"""
sweep_collect_results.py

用途：
1) 自动遍历多组超参数，分别训练 regression / classification 两种模式。
2) 训练完成后读取保存的权重（model.npz）并在本进程直接计算指标（回归：MAE/R2；分类：F1/AUC），而非从标准输出解析。
3) 生成 results.csv，并输出一份 Markdown 报告 result.md（含结果表与简要汇总）。
4) 生成若干简单曲线图（基于 Matplotlib），保存在 figures/ 目录（若有有效数据）。

使用方式：
- 建议在 LABs/LAB1 目录运行：
    python sweep_collect_results.py
- 如需在仓库根目录运行，请使用参数指定训练脚本路径：
    python LABs/LAB1/sweep_collect_results.py --train-script LABs/LAB1/train.py

先决条件：
- 可访问 train.py、submission.py、data/（相对当前工作目录或通过 --train-script 指定路径）。
- 环境已安装 numpy、pandas、matplotlib、scikit-learn、tqdm（参考项目 README）。

注意：
- 本脚本通过 subprocess 调用 `python train.py ...`，请确认你在项目根目录运行。
- 若要加/改搜索网格，请在 `SEARCH_SPACE` 中编辑。
"""

import os
import sys
import time
import json
import itertools
import subprocess
from pathlib import Path
import shutil
import argparse
from importlib import import_module
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, roc_auc_score

from data_utils import DataSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------
# 搜索网格
# ---------------------------------------------
SEARCH_SPACE = {
    "regression": {
        "seed": [42],
        "lr": [1e-3, 2e-3, 5e-3],
        "batch_size": [256, 512],
        "num_epochs": [10, 30],
        "eval_strategy": ["step"],
        "eval_steps": [200],
    },
    "classification": {
        "seed": [42],
        "lr": [5e-4, 1e-3, 2e-3],
        "batch_size": [256, 512],
        "num_epochs": [20, 40],
        "eval_strategy": ["epoch"],
        "eval_steps": [100],
    }
}

# 运行器设置（可通过命令行覆盖）
PYTHON_BIN = sys.executable  # 默认使用当前解释器；若提供 --conda-env，则优先使用 conda run
TRAIN_SCRIPT = "train.py"     # 训练脚本名

# 输出目录
OUT_DIR = Path("sweep_outputs")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

def _python_cmd(conda_env: str|None):
    """返回用于执行 train.py 的前缀命令列表。
    - 若 conda_env 不为空，且系统存在 conda，可使用：['conda','run','-n',conda_env,'python']
    - 否则回退到当前解释器 sys.executable
    """
    if conda_env:
        conda_exe = shutil.which('conda') or os.environ.get('CONDA_EXE')
        if conda_exe:
            # 在 Windows/Unix 下都可工作
            return [conda_exe, 'run', '-n', conda_env, 'python']
        else:
            print(f"[WARN] 未找到 'conda' 命令，回退使用当前解释器：{PYTHON_BIN}")
    return [PYTHON_BIN]

def _import_submission():
    """
    动态导入 submission.py 所在模块（假定与本脚本同一工程根目录）。
    """
    try:
        subm = import_module("submission")
        return subm
    except Exception as e:
        print(f"[EVAL] 无法导入 submission.py: {e}")
        return None

def _load_datasets(subm, train_path="data/train.csv", eval_path="data/valid.csv"):
    """
    为与训练时的标准化保持一致：先加载训练集（拟合 mean/std），再加载验证集（复用同一统计量）。
    submission.load_and_preprocess_data 使用模块级 mean/std 缓存。
    """
    Xtr, ytr = subm.load_and_preprocess_data(train_path)
    Xev, yev = subm.load_and_preprocess_data(eval_path)
    return (Xtr, ytr), (Xev, yev)

def _load_model_npz(npz_path):
    """
    读取保存的权重文件（默认应为 save_dir/model.npz），返回 (W, b)。
    """
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"未找到模型权重：{npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    W = data["weight"]
    b = data["bias"]
    return W, b

def _evaluate_regression(W, b, subm, Xev, yev):
    """
    评估回归：返回 (mae, r2, mae_analytic, r2_analytic)
    - analytic 部分默认在“验证集”上计算
    """
    in_features = Xev.shape[1]
    model = subm.LinearRegressionModel(in_features, 1)
    model.weight = np.asarray(W, dtype=np.float64)
    model.bias   = np.asarray(b, dtype=np.float64).reshape(-1)

    preds = model.forward(Xev).reshape(-1, 1)
    yev = yev.reshape(-1, 1)
    mae = float(mean_absolute_error(yev, preds))
    r2  = float(r2_score(yev, preds))

    # analytic on eval (与原报告口径一致)
    Wa, ba = subm.linear_regression_analytic(Xev, yev)
    model_a = subm.LinearRegressionModel(in_features, 1)
    model_a.weight = np.asarray(Wa, dtype=np.float64)
    model_a.bias   = np.asarray(ba, dtype=np.float64).reshape(-1)
    pred_a = model_a.forward(Xev).reshape(-1, 1)
    mae_a = float(mean_absolute_error(yev, pred_a))
    r2_a  = float(r2_score(yev, pred_a))

    return mae, r2, mae_a, r2_a

def _evaluate_classification(W, b, subm, Xev, yev):
    """
    评估分类：返回 (f1, auc)
    - AUC 用“概率”直接计算，避免日志里二值化导致的精度损失。
    - F1 仍用阈值 0.5 的二值化。
    - 关键点：用 DataSet(..., binary_classification=True) 生成“二值标签”，避免把连续 Run_time 当成 y_true。
    """
    # 用 DataSet 的规则把连续 target -> 二值标签
    ds_eval = DataSet(Xev, yev, binary_classification=True)
    Xev_bin = ds_eval.features
    yev_bin = ds_eval.targets  # 已经是 0/1 的向量

    in_features = Xev_bin.shape[1]
    model = subm.LogisticRegressionModel(in_features, 1)
    model.weight = np.asarray(W, dtype=np.float64)
    model.bias   = np.asarray(b, dtype=np.float64).reshape(-1)

    prob = model.forward(Xev_bin).reshape(-1)  # 概率
    pred = (prob > 0.5).astype(int)            # 0/1
    y    = np.asarray(yev_bin, dtype=int).reshape(-1)

    f1  = float(f1_score(y, pred))
    auc = float(roc_auc_score(y, prob))
    return f1, auc


def run_one(mode, params, conda_env=None):
    """
    运行一组参数，返回记录字典（训练后直接读取权重评估，避免日志精度损失）。
    """
    # 先构造保存目录（供 train.py 使用）
    run_name = f"{mode}_seed{params['seed']}_lr{params['lr']}_bs{params['batch_size']}_ep{params['num_epochs']}_{params['eval_strategy']}{params['eval_steps']}"
    run_dir = OUT_DIR / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = _python_cmd(conda_env) + [
        TRAIN_SCRIPT,
        "--mode", mode,
        "--seed", str(params["seed"]),
        "--lr", str(params["lr"]),
        "--batch_size", str(params["batch_size"]),
        "--num_epochs", str(params["num_epochs"]),
        "--eval_strategy", params["eval_strategy"],
        "--eval_steps", str(params["eval_steps"]),
        "--save_dir", str(run_dir),
    ]

    print(f"\n[RUN] {mode} :: {json.dumps(params)}")
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0

    # 始终保存原始日志，便于排障
    stdout = proc.stdout
    (OUT_DIR / f"{run_name}.log").write_text(stdout, encoding="utf-8")

    # ==== 直接评估（高精度） ====
    mae = r2 = mae_analytic = r2_analytic = np.nan
    f1 = auc = np.nan

    try:
        subm = _import_submission()
        if subm is None:
            raise RuntimeError("无法导入 submission.py")

        # 保持与训练相同的标准化：先 train 后 eval
        (Xtr, ytr), (Xev, yev) = _load_datasets(subm, "data/train.csv", "data/valid.csv")

        # 读取模型参数
        model_npz = Path(run_dir) / "model.npz"
        W, B = _load_model_npz(model_npz)

        if mode == "regression":
            mae, r2, mae_analytic, r2_analytic = _evaluate_regression(W, B, subm, Xev, yev)
        else:
            f1, auc = _evaluate_classification(W, B, subm, Xev, yev)

    except Exception as e:
        print(f"[EVAL] 直接评估失败（记录为 NaN）：{e}")

    record = {
        "mode": mode,
        **params,
        "train_time_sec": round(dt, 2),
        "mae": mae,
        "r2": r2,
        "mae_analytic": mae_analytic,
        "r2_analytic": r2_analytic,
        "f1": f1,
        "auc": auc,
        "returncode": proc.returncode,
    }
    return record

def sweep_all(conda_env=None):
    """
    全量遍历搜索网格并保存 results.csv
    """
    rows = []
    for mode in ["regression", "classification"]:
        keys = list(SEARCH_SPACE[mode].keys())
        for values in itertools.product(*[SEARCH_SPACE[mode][k] for k in keys]):
            params = dict(zip(keys, values))
            rec = run_one(mode, params, conda_env=conda_env)
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    print("\n[OK] Results saved to", OUT_DIR / "results.csv")
    return df

def plot_curves(df):
    """
    生成简单的可视化曲线图到 figures/ 目录。
    - 回归：以 (num_epochs, lr) 为维度绘制 MAE / R2 对比（散点图）
    - 分类：以 (num_epochs, lr) 为维度绘制 F1 / AUC 对比（散点图）
    """
    # 仅使用 matplotlib，且每张图一个指标
    # 回归
    dfr = df[df["mode"] == "regression"].copy()
    if not dfr.empty:
        plt.figure()
        plt.title("Regression: MAE vs num_epochs (marker size ~ lr)")
        x = dfr["num_epochs"].values
        y = dfr["mae"].values
        s = (np.array(dfr["lr"].values) / dfr["lr"].min()) * 30.0
        plt.scatter(x, y, s=s)
        plt.xlabel("num_epochs")
        plt.ylabel("MAE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "regression_mae.png", dpi=150)
        plt.close()

        plt.figure()
        plt.title("Regression: R2 vs num_epochs (marker size ~ lr)")
        y2 = dfr["r2"].values
        plt.scatter(x, y2, s=s)
        plt.xlabel("num_epochs")
        plt.ylabel("R2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "regression_r2.png", dpi=150)
        plt.close()

    # 分类
    dfc = df[df["mode"] == "classification"].copy()
    if not dfc.empty:
        plt.figure()
        plt.title("Classification: F1 vs num_epochs (marker size ~ lr)")
        x = dfc["num_epochs"].values
        y = dfc["f1"].values
        s = (np.array(dfc["lr"].values) / dfc["lr"].min()) * 30.0
        plt.scatter(x, y, s=s)
        plt.xlabel("num_epochs")
        plt.ylabel("F1")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "classification_f1.png", dpi=150)
        plt.close()

        plt.figure()
        plt.title("Classification: AUC vs num_epochs (marker size ~ lr)")
        y2 = dfc["auc"].values
        plt.scatter(x, y2, s=s)
        plt.xlabel("num_epochs")
        plt.ylabel("AUC")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "classification_auc.png", dpi=150)
        plt.close()

def best_rows(df):
    """
    选取每个任务的最优行（回归取 R2 最大、MAE 最小综合；分类取 F1 最大、AUC 最大综合）。
    """
    best = {}
    dfr = df[df["mode"] == "regression"].copy()
    if not dfr.empty:
        # 先按 R2 降序，再按 MAE 升序
        dfr = dfr.sort_values(by=["r2", "mae"], ascending=[False, True])
        best["regression"] = dfr.head(1)

    dfc = df[df["mode"] == "classification"].copy()
    if not dfc.empty:
        # 先按 F1 降序，再按 AUC 降序
        dfc = dfc.sort_values(by=["f1", "auc"], ascending=[False, False])
        best["classification"] = dfc.head(1)

    return best

def make_report(df):
    """
    生成报告 Markdown，并嵌入结果表格/图片。
    """
    report_path = OUT_DIR / "result.md"

    best = best_rows(df)
    lines = []
    lines.append("# 运行结果\n")
    
    # 超参数调整过程与现象
    lines.append("搜索网格：\n")
    lines.append("```json\n" + json.dumps(SEARCH_SPACE, ensure_ascii=False, indent=2) + "\n```\n")
    lines.append("完整结果表见 `results.csv`\n")
    # 可视化
    lines.append("\n## 结果可视化\n")
    if (FIG_DIR / "regression_mae.png").exists():
        lines.append("![reg_mae](figures/regression_mae.png)\n")
    if (FIG_DIR / "regression_r2.png").exists():
        lines.append("![reg_r2](figures/regression_r2.png)\n")
    if (FIG_DIR / "classification_f1.png").exists():
        lines.append("![cls_f1](figures/classification_f1.png)\n")
    if (FIG_DIR / "classification_auc.png").exists():
        lines.append("![cls_auc](figures/classification_auc.png)\n")

    # 最佳结果与分析
    lines.append("## 最佳结果与上限分析\n")
    if "regression" in best:
        br = best["regression"].iloc[0]
        lines.append(f"- 回归最佳：R2={br['r2']:.4f}, MAE={br['mae']:.4f}（lr={br['lr']}, bs={int(br['batch_size'])}, ep={int(br['num_epochs'])}）\n")
        if not np.isnan(br["r2_analytic"]):
            lines.append(f"  - 解析解在相同验证集上的结果：R2={br['r2_analytic']:.4f}, MAE={br['mae_analytic']:.4f}\n")
    if "classification" in best:
        bc = best["classification"].iloc[0]
        lines.append(f"- 分类最佳：F1={bc['f1']:.4f}, AUC={bc['auc']:.4f}（lr={bc['lr']}, bs={int(bc['batch_size'])}, ep={int(bc['num_epochs'])}）\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] Report markdown saved to", report_path)

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conda-env', type=str, default=None, help='指定在该 conda 环境中运行 train.py，例如 ai25')
    parser.add_argument('--train-script', type=str, default='train.py', help='可自定义 train.py 路径')
    return parser.parse_args()


def main():
    args = parse_cli()
    global TRAIN_SCRIPT
    TRAIN_SCRIPT = args.train_script

    df = sweep_all(conda_env=args.conda_env)
    plot_curves(df)
    make_report(df)
    print("\n=== Done ===")
    print(f"结果目录：{OUT_DIR.resolve()}")
    
if __name__ == "__main__":
    main()
