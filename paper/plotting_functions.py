import numpy as np
import matplotlib.pyplot as plt

def fit_loglog(x, y):
    mask = np.isfinite(y)
    b, c = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return b, np.exp(c)
def fit_loglog_last(x, y):
    mask = np.isfinite(y)
    x, y = x[mask].values[-2:], y[mask].values[-2:]
    b, c = np.polyfit(np.log(x), np.log(y), 1)
    return b, np.exp(c)
fit_loglog = fit_loglog_last

def plot_profile(dfs, skip_first=0, name_map=None, title="Top-k ClusterAttention latency breakdown"):
    plt.figure()
    col = "timings_profile"
    for df_name, df in dfs.items():
        df = df.iloc[skip_first:]
        x = df["n_tokens"]
        all_keys = sorted(df.iloc[0][col].keys())
        for name in all_keys:
            if name_map:
                if name not in name_map:
                    print(name)
                    continue
                label = name_map[name]
                linestyle = "-" if name == "forward" else "--"
            else:
                label = name
                linestyle = "--"
            label = name_map[name]
            linestyle = "-" if name == "forward" else "--"
            y = df[col].apply(lambda t, n=name: t[n])
            b, k = fit_loglog(x, y)
            if len(dfs) > 1:
                label = f"{df_name} | {label}"
            label += f" (exp coef={b:.2g})"
            plt.plot(x, y, marker="o", linestyle=linestyle, label=label)
            plt.plot(x, k * x**b, linestyle="--", color="grey", alpha=0.7)

    plt.grid(True, which='major', ls='--', alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of tokens")
    plt.ylabel("Latency (ms)")
    plt.title(title)
    plt.legend()
    plt.show()

def crossover_time(group, threshold):
    #print(group)
    x = group["time"].to_numpy()
    y = group["metric"].to_numpy()
    order = x.argsort()
    x, y = x[order], y[order]
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if min(y0, y1) <= threshold <= max(y0, y1):
            if y1 == y0:
                return x[i]
            return x[i] + (threshold - y0) / (y1 - y0) * (x[i + 1] - x[i])
    return None
def plot_pareto(
    dfs, 
    show_annotations=True, 
    show_legend=True, 
    threshold=0.99, 
    name_map=None,
    time_col=("time_ms", "Time (ms)"),
    metric_col=("cosine_sim", "Cosine similarity"),
    method_colors=None,
    figsize=(4, 4),
    logx=False
):
    method_colors = method_colors or {}
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    for df_name, df in dfs.items():
        for method, raw_group in df.groupby("method"):
            if method not in name_map.keys():
                continue
            group = raw_group.groupby("param", dropna=False, as_index=False).agg(
                time=(time_col[0], "mean"),
                metric=(metric_col[0], "mean"),
            )
            label = name_map[method]
            if len(dfs) > 1:
                label = f"{df_name} | {label}"
            if threshold is not None:
                t_cross = crossover_time(group, threshold)
                if t_cross is not None:
                    print(f"{label}: {t_cross:.1f} time units @ {threshold} metric units")
            alpha = (1 if method in ("dense", "dense_sage") else 0.5)
            zorder = (1 if method in ("dense", "dense_sage") else 0)
            line, = ax1.plot(group["time"], group["metric"], "-", label=label, alpha=alpha, lw=5, zorder=zorder, color=method_colors.get(method))
            ax1.plot(group["time"], group["metric"], "o", color=line.get_color(), alpha=0.75, zorder=line.get_zorder()+1)
            ax1.plot(
                raw_group[time_col[0]], raw_group[metric_col[0]], "o", 
                color=line.get_color(), ms=2.5, alpha=0.25, zorder=line.get_zorder()-1
                #color="black", ms=2, alpha=1, zorder=100
            )
            if show_annotations:
                for i, (_, row) in enumerate(group.iterrows()):
                    if row["param"] is None:
                        continue
                    txt = "" if method in ("dense", "dense_sage") else f"{row['param']:.2g}"
                    color = "white" if method in ("dense", "dense_sage") else "gray"
                    ax1.annotate(txt, (row["time"], row["metric"]), textcoords="offset points", xytext=(-40, 12), fontsize=6,
                                 arrowprops=dict(arrowstyle="-", lw=0.5, color=color))
    if threshold is not None:
        ax1.axhline(threshold, ls="--", color="grey", lw=1, zorder=0)
    ax1.set_xlabel(time_col[1])
    ax1.set_ylabel(metric_col[1])
    if show_legend:
        ax1.legend()
    ax1.set_axisbelow(True)
    ax1.grid(True, which='major', ls='-', alpha=0.5)
    #fig.tight_layout()
    if logx:
        ax1.set_xscale('log')
    plt.show()
