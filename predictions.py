"""
Farm-level predictive analytics module.
Generates forecasts for yield, profit, and risk from cleaned spray, weather, and financial data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_cleaned_outputs(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict:
    """Load all cleaned CSV files from the outputs directory."""
    output_path = Path(output_dir)
    return {
        "spray_logs": pd.read_csv(output_path / "cleaned_spray_logs.csv"),
        "weather": pd.read_csv(output_path / "cleaned_weather.csv"),
        "financials": pd.read_csv(output_path / "cleaned_financials.csv"),
        "metrics": pd.read_csv(output_path / "final_metrics.csv"),
    }


def predict_farm_yield(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict average yield per farm and field based on historical spray events.
    Groups by farm_id and field_id, computing mean yield_kg_per_ha.
    """
    # Use field_id_x from merged metrics (from spray logs side)
    metrics_df = metrics_df.copy()
    if "field_id_x" in metrics_df.columns and "field_id" not in metrics_df.columns:
        metrics_df["field_id"] = metrics_df["field_id_x"]
    farm_yield = (
        metrics_df.groupby(["farm_id", "field_id", "crop"])
        .agg(
            avg_yield_kg_per_ha=("yield_kg_per_ha", "mean"),
            std_yield_kg_per_ha=("yield_kg_per_ha", "std"),
            event_count=("log_id", "count"),
        )
        .reset_index()
    )
    farm_yield["std_yield_kg_per_ha"] = farm_yield["std_yield_kg_per_ha"].fillna(0)
    return farm_yield


def predict_farm_profit(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict average profit margin per farm and field.
    Groups by farm_id and field_id, computing mean profit_margin and risk metrics.
    """
    # Use field_id_x from merged metrics (from spray logs side)
    metrics_df = metrics_df.copy()
    if "field_id_x" in metrics_df.columns and "field_id" not in metrics_df.columns:
        metrics_df["field_id"] = metrics_df["field_id_x"]
    farm_profit = (
        metrics_df.groupby(["farm_id", "field_id", "crop"])
        .agg(
            avg_profit_margin=("profit_margin", "mean"),
            avg_wind_waste_proxy=("wind_waste_proxy", "mean"),
            avg_dosage_l_per_ha=("dosage_l_per_ha", "mean"),
            avg_cost_per_spray=("cost_per_spray", "mean"),
        )
        .reset_index()
    )
    return farm_profit


def predict_farm_risk(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assess farm-level risk based on wind waste and profit volatility.
    Computes wind waste proxy variance and profit margin variance.
    """
    # Use field_id_x from merged metrics (from spray logs side)
    metrics_df = metrics_df.copy()
    if "field_id_x" in metrics_df.columns and "field_id" not in metrics_df.columns:
        metrics_df["field_id"] = metrics_df["field_id_x"]
    farm_risk = (
        metrics_df.groupby(["farm_id", "field_id"])
        .agg(
            wind_waste_volatility=("wind_waste_proxy", "std"),
            profit_margin_volatility=("profit_margin", "std"),
            max_wind_waste=("wind_waste_proxy", "max"),
            min_profit_margin=("profit_margin", "min"),
        )
        .reset_index()
    )
    farm_risk["wind_waste_volatility"] = farm_risk["wind_waste_volatility"].fillna(0)
    farm_risk["profit_margin_volatility"] = farm_risk[
        "profit_margin_volatility"
    ].fillna(0)

    # Compute a simple risk score: higher wind volatility and lower min profit = higher risk.
    farm_risk["risk_score"] = (
        farm_risk["wind_waste_volatility"]
        + (1 - farm_risk["min_profit_margin"].clip(lower=0))
    )
    return farm_risk


def predict_optimal_dosage(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recommend optimal dosage per crop based on yield vs. dosage correlation.
    Finds dosage ranges that maximize yield_delta and profit_margin.
    """
    optimal = (
        metrics_df.groupby(["crop"])
        .agg(
            optimal_dosage_l_per_ha=("dosage_l_per_ha", "median"),
            median_yield_delta=("yield_delta", "median"),
            median_profit_margin=("profit_margin", "median"),
            dosage_std=("dosage_l_per_ha", "std"),
        )
        .reset_index()
    )
    optimal["dosage_std"] = optimal["dosage_std"].fillna(0.5)
    return optimal


def predict_equipment_performance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank equipment by average profit margin and reliability (low variance).
    """
    equipment = (
        metrics_df.groupby(["equipment"])
        .agg(
            avg_profit_margin=("profit_margin", "mean"),
            profit_margin_std=("profit_margin", "std"),
            avg_wind_waste=("wind_waste_proxy", "mean"),
            event_count=("log_id", "count"),
        )
        .reset_index()
    )
    equipment["profit_margin_std"] = equipment["profit_margin_std"].fillna(0)
    equipment["reliability_score"] = (
        equipment["avg_profit_margin"] / (1 + equipment["profit_margin_std"])
    )
    equipment = equipment.sort_values("reliability_score", ascending=False)
    return equipment


def predict_crop_performance(
    metrics_df: pd.DataFrame, financials_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare crop profitability across all farms.
    Combines yield delta, profit margin, and market price.
    """
    crop_perf = (
        metrics_df.groupby(["crop"])
        .agg(
            avg_yield_delta=("yield_delta", "mean"),
            avg_profit_margin=("profit_margin", "mean"),
            avg_wind_waste=("wind_waste_proxy", "mean"),
            event_count=("log_id", "count"),
        )
        .reset_index()
    )

    # Merge with financials to get pricing
    crop_perf = crop_perf.merge(
        financials_df[["crop", "price_usd_per_kg", "baseline_yield_kg_per_ha"]],
        on="crop",
        how="left",
    )
    crop_perf["profitability_score"] = (
        crop_perf["avg_yield_delta"] * crop_perf["price_usd_per_kg"]
        + crop_perf["avg_profit_margin"]
    )
    return crop_perf.sort_values("profitability_score", ascending=False)


def generate_farm_forecast(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Master function to generate all predictions and return as a dictionary.
    Each key is a forecast type; values are DataFrames.
    """
    data = load_cleaned_outputs(output_dir)
    metrics_df = data["metrics"]
    financials_df = data["financials"]
    
    # Standardize field_id naming
    metrics_df = metrics_df.copy()
    if "field_id_x" in metrics_df.columns and "field_id" not in metrics_df.columns:
        metrics_df["field_id"] = metrics_df["field_id_x"]

    forecasts = {
        "farm_yield": predict_farm_yield(metrics_df),
        "farm_profit": predict_farm_profit(metrics_df),
        "farm_risk": predict_farm_risk(metrics_df),
        "optimal_dosage": predict_optimal_dosage(metrics_df),
        "equipment_performance": predict_equipment_performance(metrics_df),
        "crop_performance": predict_crop_performance(metrics_df, financials_df),
    }

    return forecasts


def save_forecasts(
    forecasts: dict,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Save all forecast DataFrames to CSV files in a predictions subdirectory.
    Returns a dict of file paths.
    """
    output_path = Path(output_dir)
    predictions_dir = output_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in forecasts.items():
        filepath = predictions_dir / f"{name}.csv"
        df.to_csv(filepath, index=False)
        paths[name] = filepath

    return paths


if __name__ == "__main__":
    forecasts = generate_farm_forecast()

    # Print summaries
    print("=" * 80)
    print("FARM YIELD PREDICTIONS")
    print("=" * 80)
    print(forecasts["farm_yield"].head(10))

    print("\n" + "=" * 80)
    print("FARM PROFIT PREDICTIONS")
    print("=" * 80)
    print(forecasts["farm_profit"].head(10))

    print("\n" + "=" * 80)
    print("FARM RISK ASSESSMENT")
    print("=" * 80)
    print(forecasts["farm_risk"].head(10))

    print("\n" + "=" * 80)
    print("OPTIMAL DOSAGE RECOMMENDATIONS")
    print("=" * 80)
    print(forecasts["optimal_dosage"])

    print("\n" + "=" * 80)
    print("EQUIPMENT PERFORMANCE RANKING")
    print("=" * 80)
    print(forecasts["equipment_performance"])

    print("\n" + "=" * 80)
    print("CROP PROFITABILITY RANKING")
    print("=" * 80)
    print(forecasts["crop_performance"])

    # Save predictions
    paths = save_forecasts(forecasts)
    print("\n" + "=" * 80)
    print("PREDICTIONS SAVED")
    print("=" * 80)
    for name, path in paths.items():
        print(f"{name}: {path}")
