"""
Visualization module for farm analytics.
Generates farmer-friendly charts from predictions and metrics data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for professional-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_VIZ_DIR = DEFAULT_OUTPUT_DIR / "visualizations"


def load_data(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict:
    """Load all CSV files for visualization."""
    output_path = Path(output_dir)
    return {
        "metrics": pd.read_csv(output_path / "final_metrics.csv"),
        "farm_yield": pd.read_csv(output_path / "predictions" / "farm_yield.csv"),
        "farm_profit": pd.read_csv(output_path / "predictions" / "farm_profit.csv"),
        "farm_risk": pd.read_csv(output_path / "predictions" / "farm_risk.csv"),
        "equipment_performance": pd.read_csv(output_path / "predictions" / "equipment_performance.csv"),
        "crop_performance": pd.read_csv(output_path / "predictions" / "crop_performance.csv"),
        "optimal_dosage": pd.read_csv(output_path / "predictions" / "optimal_dosage.csv"),
    }


def viz_farm_yield_comparison(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Compare average yield across farms and crops."""
    fig, ax = plt.subplots(figsize=(14, 8))
    farm_yield = data["farm_yield"]
    
    # Pivot for grouped bar chart
    pivot = farm_yield.pivot_table(
        index=["farm_id", "field_id"],
        columns="crop",
        values="avg_yield_kg_per_ha"
    ).reset_index()
    
    x = np.arange(len(pivot))
    width = 0.2
    crops = ["barley", "corn", "soy", "wheat"]
    
    for i, crop in enumerate(crops):
        if crop in pivot.columns:
            ax.bar(x + i * width, pivot[crop], width, label=crop)
    
    ax.set_xlabel("Farm / Field", fontsize=12, fontweight="bold")
    ax.set_ylabel("Yield (kg/ha)", fontsize=12, fontweight="bold")
    ax.set_title("Farm Yield Comparison by Crop", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{r['farm_id']}-{r['field_id']}" for _, r in pivot.iterrows()], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    filepath = viz_dir / "01_farm_yield_comparison.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_profit_margin_by_farm(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Show profit margin distribution by farm."""
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics = data["metrics"]
    
    # Handle field_id naming
    metrics = metrics.copy()
    if "field_id_x" in metrics.columns:
        metrics["field_id"] = metrics["field_id_x"]
    
    # Create farm-field labels
    metrics["farm_field"] = metrics["farm_id"] + "-" + metrics["field_id"]
    
    sns.boxplot(data=metrics, x="farm_field", y="profit_margin", ax=ax)
    ax.set_xlabel("Farm / Field", fontsize=12, fontweight="bold")
    ax.set_ylabel("Profit Margin", fontsize=12, fontweight="bold")
    ax.set_title("Profit Margin Distribution by Farm & Field", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    filepath = viz_dir / "02_profit_margin_boxplot.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_risk_assessment_heatmap(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Heatmap of farm risk scores."""
    fig, ax = plt.subplots(figsize=(10, 8))
    farm_risk = data["farm_risk"].copy()
    farm_risk["farm_field"] = farm_risk["farm_id"] + "-" + farm_risk["field_id"]
    
    pivot = farm_risk.pivot_table(
        index="field_id", columns="farm_id", values="risk_score"
    )
    
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Risk Score"})
    ax.set_title("Farm Risk Assessment Heatmap\n(Higher = More Risk)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Farm", fontsize=12, fontweight="bold")
    ax.set_ylabel("Field", fontsize=12, fontweight="bold")
    
    filepath = viz_dir / "03_risk_assessment_heatmap.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_equipment_performance(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Equipment ranking by profitability and consistency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    equipment = data["equipment_performance"].sort_values("reliability_score", ascending=True)
    
    # Reliability score
    ax1.barh(equipment["equipment"], equipment["reliability_score"], color="steelblue")
    ax1.set_xlabel("Reliability Score\n(profit margin / volatility)", fontsize=11, fontweight="bold")
    ax1.set_title("Equipment Reliability Ranking", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    
    # Avg wind waste
    equipment_sorted = equipment.sort_values("avg_wind_waste", ascending=True)
    ax2.barh(equipment_sorted["equipment"], equipment_sorted["avg_wind_waste"], color="coral")
    ax2.set_xlabel("Average Wind Waste Proxy", fontsize=11, fontweight="bold")
    ax2.set_title("Equipment Wind Waste Performance", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    
    fig.suptitle("Equipment Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    filepath = viz_dir / "04_equipment_performance.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_crop_profitability(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Crop profitability ranking."""
    fig, ax = plt.subplots(figsize=(12, 7))
    crop_perf = data["crop_performance"].sort_values("profitability_score", ascending=True)
    
    colors = ["red" if x < 0 else "green" for x in crop_perf["profitability_score"]]
    ax.barh(crop_perf["crop"], crop_perf["profitability_score"], color=colors, alpha=0.7)
    ax.set_xlabel("Profitability Score\n(yield delta * price + profit margin)", fontsize=11, fontweight="bold")
    ax.set_title("Crop Profitability Ranking", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(crop_perf["profitability_score"]):
        ax.text(v + 50, i, f"{v:.0f}", va="center", fontweight="bold")
    
    filepath = viz_dir / "05_crop_profitability_ranking.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_wind_waste_vs_profit(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Scatter plot: wind waste proxy vs profit margin."""
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = data["metrics"]
    
    scatter = ax.scatter(
        metrics["wind_waste_proxy"],
        metrics["profit_margin"],
        c=metrics["wind_speed_mps"],
        s=100,
        alpha=0.6,
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5
    )
    
    ax.set_xlabel("Wind Waste Proxy (wind speed × dosage)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Profit Margin", fontsize=12, fontweight="bold")
    ax.set_title("Wind Waste vs Profit Margin\n(Color = Wind Speed)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Wind Speed (m/s)", fontsize=11, fontweight="bold")
    
    # Add trend line (handle NaN values consistently)
    valid_mask = metrics["wind_waste_proxy"].notna() & metrics["profit_margin"].notna()
    if valid_mask.sum() > 1:
        z = np.polyfit(metrics.loc[valid_mask, "wind_waste_proxy"], metrics.loc[valid_mask, "profit_margin"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(metrics["wind_waste_proxy"].min(), metrics["wind_waste_proxy"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend", alpha=0.8)
    ax.legend()
    
    filepath = viz_dir / "06_wind_waste_vs_profit.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_dosage_recommendations(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Optimal dosage by crop with confidence bands."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    optimal = data["optimal_dosage"].sort_values("optimal_dosage_l_per_ha")
    
    # Dosage recommendations
    colors = ax1.bar(optimal["crop"], optimal["optimal_dosage_l_per_ha"], color="steelblue", alpha=0.7)
    ax1.errorbar(optimal["crop"], optimal["optimal_dosage_l_per_ha"], 
                 yerr=optimal["dosage_std"], fmt="none", color="black", capsize=5, capthick=2)
    ax1.set_ylabel("Optimal Dosage (l/ha)", fontsize=11, fontweight="bold")
    ax1.set_title("Optimal Dosage by Crop\n(with std deviation)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    
    # Median profit margin by crop
    optimal_sorted = optimal.sort_values("median_profit_margin", ascending=True)
    ax2.barh(optimal_sorted["crop"], optimal_sorted["median_profit_margin"], color="coral", alpha=0.7)
    ax2.set_xlabel("Median Profit Margin", fontsize=11, fontweight="bold")
    ax2.set_title("Profit Margin by Crop\n(at optimal dosage)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    
    fig.suptitle("Dosage Optimization Recommendations", fontsize=14, fontweight="bold", y=1.02)
    filepath = viz_dir / "07_dosage_recommendations.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_yield_delta_distribution(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Distribution of yield delta (actual vs baseline)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics = data["metrics"]
    
    # Histogram by crop
    crops = metrics["crop"].unique()
    for crop in sorted(crops):
        crop_data = metrics[metrics["crop"] == crop]["yield_delta"]
        ax.hist(crop_data, bins=20, alpha=0.6, label=crop)
    
    ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Baseline")
    ax.set_xlabel("Yield Delta (kg/ha above/below baseline)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("Yield Delta Distribution by Crop\n(Negative = Below Baseline)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    filepath = viz_dir / "08_yield_delta_distribution.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_farm_profit_summary(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Farm-level profit summary with dosage and wind waste."""
    fig, ax = plt.subplots(figsize=(14, 8))
    farm_profit = data["farm_profit"]
    
    farm_profit_agg = farm_profit.groupby("farm_id").agg({
        "avg_profit_margin": "mean",
        "avg_wind_waste_proxy": "mean",
        "avg_dosage_l_per_ha": "mean",
        "avg_cost_per_spray": "mean"
    }).reset_index()
    
    x = np.arange(len(farm_profit_agg))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width / 2, farm_profit_agg["avg_profit_margin"], width, 
                   label="Avg Profit Margin", color="steelblue", alpha=0.7)
    bars2 = ax2.bar(x + width / 2, farm_profit_agg["avg_wind_waste_proxy"], width,
                    label="Avg Wind Waste", color="coral", alpha=0.7)
    
    ax.set_xlabel("Farm", fontsize=12, fontweight="bold")
    ax.set_ylabel("Profit Margin", fontsize=11, fontweight="bold", color="steelblue")
    ax2.set_ylabel("Wind Waste Proxy", fontsize=11, fontweight="bold", color="coral")
    ax.set_title("Farm Profit vs Wind Waste Summary", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(farm_profit_agg["farm_id"])
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax.grid(axis="y", alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    filepath = viz_dir / "09_farm_profit_summary.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def viz_temperature_impact_on_yield(data: dict, viz_dir: Path = DEFAULT_VIZ_DIR) -> Path:
    """Scatter: temperature vs yield to show weather impact."""
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = data["metrics"].dropna(subset=["temperature_c", "yield_kg_per_ha"])
    
    scatter = ax.scatter(
        metrics["temperature_c"],
        metrics["yield_kg_per_ha"],
        c=metrics["humidity"],
        s=100,
        alpha=0.6,
        cmap="coolwarm",
        edgecolors="black",
        linewidth=0.5
    )
    
    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Yield (kg/ha)", fontsize=12, fontweight="bold")
    ax.set_title("Weather Impact on Yield\n(Color = Humidity)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Humidity", fontsize=11, fontweight="bold")
    
    filepath = viz_dir / "10_temperature_impact_on_yield.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def generate_all_visualizations(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    viz_dir: Path | str = DEFAULT_VIZ_DIR,
) -> dict:
    """Generate all visualizations and return file paths."""
    viz_path = Path(viz_dir)
    viz_path.mkdir(parents=True, exist_ok=True)
    
    data = load_data(output_dir)
    
    print("Generating visualizations...")
    paths = {
        "farm_yield_comparison": viz_farm_yield_comparison(data, viz_path),
        "profit_margin_boxplot": viz_profit_margin_by_farm(data, viz_path),
        "risk_assessment_heatmap": viz_risk_assessment_heatmap(data, viz_path),
        "equipment_performance": viz_equipment_performance(data, viz_path),
        "crop_profitability": viz_crop_profitability(data, viz_path),
        "wind_waste_vs_profit": viz_wind_waste_vs_profit(data, viz_path),
        "dosage_recommendations": viz_dosage_recommendations(data, viz_path),
        "yield_delta_distribution": viz_yield_delta_distribution(data, viz_path),
        "farm_profit_summary": viz_farm_profit_summary(data, viz_path),
        "temperature_impact_on_yield": viz_temperature_impact_on_yield(data, viz_path),
    }
    
    return paths


if __name__ == "__main__":
    paths = generate_all_visualizations()
    print("\n" + "=" * 80)
    print("VISUALIZATIONS GENERATED")
    print("=" * 80)
    for name, path in paths.items():
        print(f"{name}: {path}")
