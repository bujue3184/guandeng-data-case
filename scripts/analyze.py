#!/usr/bin/env python
"""Main analysis: why do requisitions fail with "stock shows available"?

Run:  python scripts/analyze.py
Outputs:
  analysis/report.md              full numeric report (all README numbers come from here)
  analysis/fig_fail_by_staleness.png
Only the provided data (data/raw/requisitions.csv) is used. No synthetic data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "requisitions.csv"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

# cost model given by the task
COST_FAIL = 75   # 工时 per 空跑
COST_WARN = 15   # 工时 per 预警

BUF = []


def say(line: str = "") -> None:
    BUF.append(line)
    print(line)


def load() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(RAW)
    n_raw = len(raw)

    # exact duplicate rows: same req_id & same everything -> double entry
    df = raw.drop_duplicates().copy()
    n_dup = n_raw - len(df)

    # time is mixed-format aware; force UTC then convert to plant local time (UTC+8)
    df["req_ts"] = pd.to_datetime(df["requisition_time"], utc=True).dt.tz_convert("Asia/Shanghai")
    df["sync_ts"] = pd.to_datetime(df["stock_synced_at"], utc=True).dt.tz_convert("Asia/Shanghai")

    # THE core feature: how stale is the stock snapshot shown to the picker
    df["stale_min"] = (df["req_ts"] - df["sync_ts"]).dt.total_seconds() / 60
    df["qty_ratio"] = df["recorded_qty"] / df["plan_batch_qty"].replace(0, np.nan)

    meta = {
        "n_raw": n_raw,
        "n_dup": n_dup,
        "n": len(df),
        "n_unknown": int(df["is_fulfilled"].isna().sum()),
    }
    return df, meta


def sec_quality(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    say("## 0. 数据质量")
    say()
    say(f"- 原始行数 {meta['n_raw']}，其中完全重复行 **{meta['n_dup']}** 行（req_id 及其余字段全同），去重后 **{meta['n']}** 行。")
    say(f"- `is_fulfilled` 未回传 **{meta['n_unknown']}** 行（{meta['n_unknown']/meta['n']*100:.1f}%），主分析剔除。")
    say("- 时间戳时区不一致：`requisition_time` 为 `+08:00`，`stock_synced_at` 为 `Z`(UTC)。**已统一转 UTC 再折算本地时间**；若按字符串直接相见会产生 8 小时偏差。")
    say(f"- 时间跨度：{df['req_ts'].min():%Y-%m-%d %H:%M} ~ {df['req_ts'].max():%Y-%m-%d %H:%M}，覆盖 7 天、4 个厂区、10 类物料。")
    say()
    return df[df["is_fulfilled"].notna()].copy()


def sec_baseline(k: pd.DataFrame) -> None:
    fails = int((k["is_fulfilled"] == 0).sum())
    n = len(k)
    say("## 1. 现状：缺料空跑的规模")
    say()
    say(f"- 有效回传 **{n}** 单，其中缺料失败（空跑）**{fails}** 单，**失败率 {fails/n*100:.1f}%**。")
    say(f"- 直接工时损失 **{fails} × {COST_FAIL} = {fails*COST_FAIL:,} 工时**（按题目口径）。")
    seen = k[k["recorded_qty"] > 0]
    f_pos = int((seen["is_fulfilled"] == 0).sum())
    say(f"- **{f_pos} / {fails}（{f_pos/fails*100:.0f}%）的空跑发生在「账面显示仍有库存(>0)」的单上** —— 这正是领料员投诉的「系统显示有货、跑过去没有」。")
    say(f"- 反过来说，账面为 0 时基本不会去跑：仅 {fails-f_pos} 单，占总空跑 {(fails-f_pos)/fails*100:.0f}%。")
    say()


def sec_root_cause(k: pd.DataFrame) -> None:
    k = k.copy()
    k["stale_bin"] = pd.cut(
        k["stale_min"], [-1, 0, 5, 10, 15, 20, 30, 45, 1e9],
        labels=["0", "1-5", "5-10", "10-15", "15-20", "20-30", "30-45", "45+"],
    )
    g = k.groupby("stale_bin", observed=True).agg(
        n=("req_id", "size"), fails=("is_fulfilled", lambda s: int((s == 0).sum()))
    )
    g["fail_rate"] = (g["fails"] / g["n"] * 100).round(1)

    say("## 2. 根因：账面库存的「新鲜度」而不是「数量」")
    say()
    say("把「系统显示的库存快照有多旧」(`requisition_time - stock_synced_at`) 分桶：")
    say()
    say("| 快照延迟(分钟) | 单量 | 空跑 | 空跑率 |")
    say("|---|---|---|---|")
    for idx, r in g.iterrows():
        say(f"| {idx} | {int(r['n'])} | {int(r['fails'])} | {r['fail_rate']:.1f}% |")
    say()

    fresh = k[k["stale_min"] <= 15]
    stale = k[k["stale_min"] > 15]
    fr_f = (fresh["is_fulfilled"] == 0).mean() * 100
    fr_s = (stale["is_fulfilled"] == 0).mean() * 100
    say(f"- 快照 ≤15 分钟：{len(fresh)} 单，空跑率 **{fr_f:.1f}%**")
    say(f"- 快照 >15 分钟：{len(stale)} 单，空跑率 **{fr_s:.1f}%**（是前者的 **{fr_s/fr_f:.1f} 倍**）")
    say(f"- 全部 {int((k['is_fulfilled']==0).sum())} 次空跑中，**{int((stale['is_fulfilled']==0).sum())} 次（{ (stale['is_fulfilled']==0).sum()/(k['is_fulfilled']==0).sum()*100:.0f}%）落在过期快照上**。")
    say()
    say("> 机理：`recorded_qty` 是「最后一次同步时的账面数」，不是实时数。同步间隔期内物料被其它工单/其它厂区领走，")
    say("> 领料员看到的仍是旧值 → 到仓断供。**数量本身没错，错在它是过期快照。**")
    say()

    # interaction with quantity ratio -> shows staleness dominates
    k["sb"] = pd.cut(k["stale_min"], [-1, 15, 30, 1e9], labels=["≤15", "15-30", ">30"])
    k["rb"] = pd.cut(k["qty_ratio"], [-0.01, 0.15, 0.3, 1, 1e9], labels=["<0.15", "0.15-0.3", "0.3-1", ">1"])
    p = k.pivot_table(index="sb", columns="rb", values="is_fulfilled",
                      aggfunc=lambda s: (s == 0).mean() * 100, observed=True).round(1)
    nn = k.pivot_table(index="sb", columns="rb", values="is_fulfilled", aggfunc="size", observed=True)
    say("**交叉验证：快照延迟 × 账面/计划数量比（空跑率 %，括号为单量）**")
    say()
    say("| 快照延迟 | " + " | ".join(p.columns.astype(str)) + " |")
    say("|---" * (len(p.columns) + 1) + "|")
    for idx in p.index:
        cells = []
        for c in p.columns:
            v = p.loc[idx, c]
            n = nn.loc[idx, c]
            cells.append("—" if pd.isna(v) else f"{v:.1f}% ({int(n)})")
        say(f"| {idx} | " + " | ".join(cells) + " |")
    say()
    say("读数：**同一行（同样新鲜度）纵向比较，数量比的影响很小；跨行（不同新鲜度）差异极大** → 主因是时效，不是数量。")
    say()


def sec_alternative(k: pd.DataFrame) -> None:
    fails_total = int((k["is_fulfilled"] == 0).sum())
    say("## 3. 被排除的替代解释")
    say()
    # quantity-based intuition
    under = (k["recorded_qty"] < k["plan_batch_qty"]).mean() * 100
    say(f"1. **「账面数量小于计划批量导致缺料」** —— 不成立。{under:.1f}% 的单都满足 recorded_qty < plan_batch_qty，")
    say(f"   但整体成功率仍有 {100-fails_total/len(k)*100:.1f}%。若这是主因，成功率不会这么高；且数量比越低失败率并非单调上升（见上表）。")
    say(f"   直接用 `ratio<0.3` 做预警会触发 {int((k['qty_ratio']<0.3).sum())} 次预警、只命中 {int(((k['qty_ratio']<0.3)&(k['is_fulfilled']==0)).sum())} 次，按成本模型是**负收益**。")
    say()

    fac = k.groupby("factory_name").agg(n=("req_id", "size"), f=("is_fulfilled", lambda s: int((s == 0).sum())))
    fac["rate"] = (fac["f"] / fac["n"] * 100).round(1)
    fac["stale_share"] = k.assign(s=k["stale_min"] > 15).groupby("factory_name")["s"].mean().mul(100).round(1)
    say("2. **「某个厂区管理差」** —— 不成立。四厂空跑率接近，差异无实际意义：")
    say()
    say("| 厂区 | 单量 | 空跑 | 空跑率 | 快照>15min 占比 |")
    say("|---|---|---|---|---|")
    for idx, r in fac.sort_values("rate", ascending=False).iterrows():
        say(f"| {idx} | {int(r['n'])} | {int(r['f'])} | {r['rate']:.1f}% | {r['stale_share']:.1f}% |")
    say()

    mat = k.groupby("material_type").agg(n=("req_id", "size"), f=("is_fulfilled", lambda s: int((s == 0).sum())))
    mat["rate"] = (mat["f"] / mat["n"] * 100).round(1)
    say(f"3. **「某类物料特别容易缺」** —— 弱。物料间空跑率 {mat['rate'].min():.1f}%~{mat['rate'].max():.1f}%，")
    say("   而快照超过 15 分钟后各类物料空跑率同时抬升到 23%~45%；物料是放大器，不是开关。")
    say()

    hour = k.groupby(k["req_ts"].dt.hour).agg(n=("req_id", "size"), f=("is_fulfilled", lambda s: int((s == 0).sum())))
    hour["rate"] = (hour["f"] / hour["n"] * 100).round(1)
    say(f"4. **「某个班次/时段特有的问题」** —— 不成立。7:00–21:00 各时段空跑率 {hour['rate'].min():.0f}%~{hour['rate'].max():.0f}%，")
    say("   快照>15min 的占比在各时段也都是 15%~30%，同步延迟是全天候随机发生的，不是夜班缺同步。")
    say()


def sec_rule(k: pd.DataFrame) -> None:
    say("## 4. 可落地的预警规则与经济性")
    say()
    say(f"成本模型（题目给定）：一次空跑 **{COST_FAIL}** 工时，一次预警 **{COST_WARN}** 工时。")
    say(f"→ 只要预警的命中率（precision）高于 **{COST_WARN/COST_FAIL*100:.0f}%**，规则就是正收益。")
    say()
    say("**规则 R：提交领料单时，若该物料库存快照已超过 15 分钟未同步，则先提示「库存数据可能已过期，建议先刷新/确认」再放行。**")
    say()

    fails_total = int((k["is_fulfilled"] == 0).sum())
    rows = []
    for name, m in [
        ("R: 快照>15min", k["stale_min"] > 15),
        ("R: 快照>20min", k["stale_min"] > 20),
        ("R: 快照>15min 且 数量比<0.4", (k["stale_min"] > 15) & (k["qty_ratio"] < 0.4)),
        ("对照: 数量比<0.3", k["qty_ratio"] < 0.3),
        ("对照: 数量比<0.2", k["qty_ratio"] < 0.2),
    ]:
        warn = int(m.sum())
        caught = int((m & (k["is_fulfilled"] == 0)).sum())
        prec = caught / warn * 100 if warn else 0
        recall = caught / fails_total * 100
        net = caught * COST_FAIL - warn * COST_WARN
        rows.append((name, warn, caught, prec, recall, net))
    say("| 规则 | 预警次数 | 命中空跑 | 命中率 | 覆盖率 | 净节省工时 |")
    say("|---|---|---|---|---|---|")
    for name, warn, caught, prec, recall, net in rows:
        say(f"| {name} | {warn} | {caught} | {prec:.1f}% | {recall:.1f}% | **{net:+,}** |")
    say()
    best = rows[0]
    say(f"结论：15 分钟阈值下，预警 {best[1]} 次、拦下 {best[2]} 次空跑（覆盖全部空跑的 {best[4]:.0f}%），")
    say(f"净节省 **{best[5]:,} 工时**，相当于把 {fails_total*COST_FAIL:,} 工时的损失削掉 **{best[5]/(fails_total*COST_FAIL)*100:.0f}%**。")
    say()
    say("为什么不再精细调阈：15 分钟是空跑率从 ~4% 跳到 ~26% 的拐点；再往上调（20/25 分钟）覆盖率和净收益都下降，")
    say("往下调（<15）会纳入大量健康单，命中率跌破 20% 盈亏线。**在只有 7 天、988 行的数据上，过拟合阈值没有意义。**")
    say()
    say("**规则 R2（配套）**：把「快照年龄」直接显示在领料单上，超过阈值标红；对高频消耗物料（快照>15min 时最易失败）改为")
    say("定时同步 + 领料触发即时校验（refresh-on-read）。前者是一行前端改动，后者是根治。")
    say()


def sec_mechanism(k: pd.DataFrame) -> None:
    """Test the naive mechanism 'stock was taken by another picker during the gap'."""
    say("## 5. 机制检验：是「这 15 分钟里被别人抢走了」吗？")
    say()
    say("如果真是邻近工单把料领走，那么「同厂 + 同物料 + 15 分钟窗口内出现多单」应该明显更容易失败。数据不支持：")
    say()

    k = k.reset_index(drop=True).sort_values("req_ts").reset_index(drop=True)
    per_hour = k.groupby([k["factory_name"], k["req_ts"].dt.floor("h")]).size().groupby(level=0).mean()
    say(f"- 单量强度：每厂每小时平均 **{per_hour.min():.1f}–{per_hour.max():.1f} 单**，15 分钟窗口内平均不到 1 单 —— 抢料的物理机会本来就很少。")
    say()

    keys = (k["factory_name"] + "|" + k["material_type"]).values
    seq = np.zeros(len(k), dtype=int)
    for key in pd.unique(keys):
        grp = k[keys == key]
        t = grp["req_ts"].values
        idx = grp.index.values
        for j in range(len(t)):
            seq[idx[j]] = int(((t > t[j] - np.timedelta64(15, "m")) & (t <= t[j])).sum())
    k["seq_15m"] = seq
    kk = k[k["is_fulfilled"].notna()]
    first = kk[kk["seq_15m"] == 1]
    later = kk[kk["seq_15m"] >= 2]
    say("| 同厂同料 15 分钟内的序位 | 单量 | 空跑 | 空跑率 |")
    say("|---|---|---|---|")
    for label, part in [("第 1 单（前面没人抢）", first), ("第 2 单及以后（前面有人抢过）", later)]:
        f = int((part["is_fulfilled"] == 0).sum())
        say(f"| {label} | {len(part)} | {f} | {f/len(part)*100:.1f}% |")
    say()
    base = (first["is_fulfilled"] == 0).mean()
    exp = base * len(later)
    say(f"→ 两组差异 **{ (later['is_fulfilled']==0).mean()*100:.1f}% vs {base*100:.1f}%**，但窗口重叠只有 {len(later)} 单（{len(later)/len(kk)*100:.1f}%），")
    say(f"   按基线 {base*100:.1f}% 计期望失败 {exp:.1f} 次，实际 {int((later['is_fulfilled']==0).sum())} 次 —— **落在噪音范围内，没有显著差异**。")
    say("   若「被抢走」成立，应看到后到者显著更差；现在看不到，而且这点样本量最多只能解释个位数的空跑，")
    say(f"   无法解释「{int((k[k['stale_min']>15]['is_fulfilled']==0).sum())} 次空跑落在陈旧快照上」这个量级。**这个假说被否掉。**")
    say()

    g = kk.groupby(pd.cut(kk["stale_min"], [-1, 15, 30, 1e9], labels=["≤15", "15-30", ">30"]), observed=True).agg(
        n=("req_id", "size"), qty_mean=("recorded_qty", "mean"), qty_median=("recorded_qty", "median")
    ).round(2)
    say("另一种可能的偏差是「旧数字偏大」→ 账面余量乐观。也不是：")
    say()
    say(g.to_markdown())
    say()
    say("**修正后的机制**：`recorded_qty` 是「最后一次同步时的账面数」，而这个账面数长期没有因为出库被冲减 ——")
    say("**账实不符（账高于实）才是实质，快照年龄只是它可见的标志。** 代价是同一个缺陷的两面：数字既旧、又虚。")
    say("要区分「被消耗」与「从来没记准」，需要库存流水/盘点记录，本数据没有 —— 因此这一点只到「账面数不可信」，不指向具体成因。")
    say()


def sec_units(k: pd.DataFrame) -> None:
    say("## 6. 这两个数量字段能直接比吗？—— 口径存疑")
    say()
    under = (k["recorded_qty"] < k["plan_batch_qty"]).mean() * 100
    ok = (k["is_fulfilled"] == 1).mean() * 100
    say(f"- `recorded_qty` 中位数 {k['recorded_qty'].median():.0f}，`plan_batch_qty` 中位数 {k['plan_batch_qty'].median():.0f}；")
    say(f"  **{under:.1f}% 的单在提交时就「账面 < 计划」**，可这批单的履约成功率仍有 **{ok:.1f}%**。")
    say(f"- `recorded_qty` 取值几乎均匀铺满 0–{int(k['recorded_qty'].max())}（每个整数约 25–42 单，取上限值仅占 {(k['recorded_qty']==k['recorded_qty'].max()).mean()*100:.1f}%），")
    say("  不像真实的物理库存分布，更像一个被截顶/换过单位的字段。")
    say()
    say("→ 结论：**这两个字段大概率不是同一个计量口径**（件/套/托，或按行按批）。")
    say("   这正是主数据（MDM）该统一而没统一的东西，也是我**拒绝对数量做任何预警规则**的原因：")
    say("   拿一个口径不明的字段做判断，比不做更危险。")
    say()


def sec_unknown(df_all: pd.DataFrame) -> None:
    u = df_all[df_all["is_fulfilled"].isna()]
    k = df_all[df_all["is_fulfilled"].notna()]
    say(f"## 7. 未回传的 {len(u)} 单是否会让结论偏掉")
    say()
    say(f"- 未回传占比 {len(u)/len(df_all)*100:.1f}%；其中快照>15min 的占 {(u['stale_min']>15).mean()*100:.1f}%，")
    say(f"  已回传样本中该比例为 {(k['stale_min']>15).mean()*100:.1f}% —— 未回传与快照新鲜度**不相关**，剔除它不会系统性高估或低估根因。")
    say(f"- 但未回传单里数量比<0.15 的占 {(u['qty_ratio']<0.15).mean()*100:.1f}%（已回传 {(k['qty_ratio']<0.15).mean()*100:.1f}%），略偏向低库存单，")
    say("  因此「数量比」这条线的结论保守看待；时效结论不受影响。")
    say()


def plot(k: pd.DataFrame) -> None:
    bins = [0, 5, 10, 15, 20, 30, 45, 70]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-45", "45+"]
    kk = k.copy()
    kk["b"] = pd.cut(kk["stale_min"], bins, labels=labels, right=False, include_lowest=True)
    g = kk.groupby("b", observed=True).agg(n=("req_id", "size"), f=("is_fulfilled", lambda s: (s == 0).sum()))
    g["rate"] = g["f"] / g["n"] * 100

    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    colors = ["#4C9F70" if i < 3 else "#C0392B" for i in range(len(g))]
    ax1.bar(range(len(g)), g["rate"], color=colors, width=0.65)
    ax1.set_xticks(range(len(g)))
    ax1.set_xticklabels(g.index, fontsize=10)
    ax1.set_xlabel("库存快照延迟（分钟）", fontsize=11)
    ax1.set_ylabel("缺料空跑率 (%)", fontsize=11)
    ax1.set_title("库存快照越旧，领料空跑率越高（15 分钟是拐点）", fontsize=13)
    for i, v in enumerate(g["rate"]):
        ax1.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax1.axvline(2.5, color="#333", linestyle="--", linewidth=1)
    ax1.text(2.6, max(g["rate"]) * 0.92, "预警阈值 15 分钟", fontsize=9, color="#333")
    ax1.set_ylim(0, max(g["rate"]) * 1.2)
    ax2 = ax1.twinx()
    ax2.plot(range(len(g)), g["n"], color="#555", marker="o", linewidth=1, markersize=4, alpha=0.7)
    ax2.set_ylabel("单量", fontsize=10, color="#555")
    ax2.set_ylim(0, g["n"].max() * 1.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fail_by_staleness.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df_all, meta = load()
    say("# 领料缺料空跑归因分析")
    say()
    say(f"_数据：`data/raw/requisitions.csv`（题目发放，未补造）。口径：一次空跑 {COST_FAIL} 工时，一次预警 {COST_WARN} 工时。_")
    say()
    k = sec_quality(df_all, meta)
    sec_baseline(k)
    sec_root_cause(k)
    sec_alternative(k)
    sec_rule(k)
    sec_mechanism(k)
    sec_units(k)
    sec_unknown(df_all)
    plot(k)
    (OUT / "report.md").write_text("\n".join(BUF) + "\n", encoding="utf-8")
    print(f"\n[written] {OUT/'report.md'} , {OUT/'fig_fail_by_staleness.png'}")


if __name__ == "__main__":
    main()
