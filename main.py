
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema


DEFAULT_SPRAY_LOGS_CSV = Path(__file__).resolve().parent / "data" / "raw" / "spray_logs_500.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


CANONICAL_LOG_COLUMNS = {
	"log_id",
	"farm_id",
	"field_id",
	"timestamp",
	"crop",
	"equipment",
	"operator_id",
	"area_ha",
	"dosage_value",
	"dosage_unit",
	"spray_cost_cents",
	"yield_kg",
}

CANONICAL_WEATHER_COLUMNS = {
	"field_id",
	"weather_time",
	"ingest_time",
	"wind_speed_mps",
	"temperature_c",
	"humidity",
}

CANONICAL_FIN_COLUMNS = {
	"crop",
	"price_usd_per_kg",
	"baseline_yield_kg_per_ha",
	"cost_per_ha",
}


# Alias maps simulate schema drift by renaming canonical fields.
LOG_ALIASES = {
	"timestamp": ["timestamp", "event_time", "logged_at"],
	"dosage_value": ["dosage_value", "dose_val", "dosage_raw"],
	"dosage_unit": ["dosage_unit", "dose_unit", "unit"],
	"spray_cost_cents": ["spray_cost_cents", "spray_cost_cent", "spray_cost"],
	"yield_kg": ["yield_kg", "yield_kgs", "yield_mass"],
}

WEATHER_ALIASES = {
	"weather_time": ["weather_time", "obs_time", "timestamp"],
	"wind_speed_mps": ["wind_speed_mps", "wind_mps", "wind_speed"],
	"temperature_c": ["temperature_c", "temp_c", "temperature"],
	"humidity": ["humidity", "rh", "rel_humidity"],
}

FIN_ALIASES = {
	"price_usd_per_kg": ["price_usd_per_kg", "price", "price_usd"],
	"baseline_yield_kg_per_ha": ["baseline_yield_kg_per_ha", "baseline_yield"],
	"cost_per_ha": ["cost_per_ha", "cost_ha", "cost"],
}


def _choose_aliases(rng: np.random.Generator, alias_map: Dict[str, Iterable[str]]) -> Dict[str, str]:
	return {key: rng.choice(list(values)) for key, values in alias_map.items()}


def _apply_aliases(df: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
	columns = {canonical: alias for canonical, alias in alias_map.items() if canonical in df.columns}
	return df.rename(columns=columns)


def synthesize_data(
	n_records: int = 500,
	seed: int = 42,
) -> Dict[str, pd.DataFrame]:
	"""
	Generate realistic spray logs with three farm profiles:
	- F-100: Struggling farm (high wind waste, poor practices, low yield)
	- F-200: Average farm (moderate performance)
	- F-300: High performer (efficient, lower costs, better yields)
	"""
	rng = np.random.default_rng(seed)
	base_time = pd.Timestamp("2025-01-01")

	# Create farm profile assignments
	farm_distribution = np.array(["F-100"] * (n_records // 3) + 
	                               ["F-200"] * (n_records // 3) + 
	                               ["F-300"] * (n_records - 2 * (n_records // 3)))
	rng.shuffle(farm_distribution)

	farm_ids = farm_distribution
	field_ids = rng.choice(["Field-A", "Field-B", "Field-C", "Field-D"], size=n_records)
	timestamps = base_time + pd.to_timedelta(rng.integers(0, 60 * 60 * 24 * 90, size=n_records), unit="s")
	crops = rng.choice(["corn", "wheat", "soy", "barley"], size=n_records)
	operator_id = rng.integers(1000, 1030, size=n_records)
	area_ha = rng.uniform(5.0, 60.0, size=n_records)

	# Farm-specific spray patterns
	dosage_value = np.zeros(n_records)
	equipment = np.zeros(n_records, dtype=object)
	spray_cost_cents = np.zeros(n_records)
	yield_kg = np.zeros(n_records)

	for i, farm_id in enumerate(farm_ids):
		if farm_id == "F-100":
			# Struggling farm: high dosage (overcompensation), expensive, low yield
			dosage_value[i] = np.clip(rng.normal(2.0, 0.5, 1)[0], 0.8, 3.5)
			equipment[i] = rng.choice(["sprayer-x", "sprayer-x", "sprayer-y"], p=[0.6, 0.3, 0.1])
			spray_cost_cents[i] = (dosage_value[i] * area_ha[i] * rng.uniform(180, 220)).astype(int)
			yield_kg[i] = (area_ha[i] * rng.uniform(1500, 2800) + rng.normal(0, 300)).astype(int)
		elif farm_id == "F-200":
			# Average farm: moderate dosage, moderate cost, moderate yield
			dosage_value[i] = np.clip(rng.normal(1.2, 0.3, 1)[0], 0.5, 2.5)
			equipment[i] = rng.choice(["sprayer-x", "sprayer-y", "sprayer-z"], p=[0.33, 0.33, 0.34])
			spray_cost_cents[i] = (dosage_value[i] * area_ha[i] * rng.uniform(140, 180)).astype(int)
			yield_kg[i] = (area_ha[i] * rng.uniform(3000, 4500) + rng.normal(0, 200)).astype(int)
		else:  # F-300
			# High performer: efficient dosage, low cost, high yield
			dosage_value[i] = np.clip(rng.normal(0.9, 0.2, 1)[0], 0.3, 1.8)
			equipment[i] = rng.choice(["sprayer-y", "sprayer-z", "sprayer-z"], p=[0.2, 0.4, 0.4])
			spray_cost_cents[i] = (dosage_value[i] * area_ha[i] * rng.uniform(100, 140)).astype(int)
			yield_kg[i] = (area_ha[i] * rng.uniform(4000, 5500) + rng.normal(0, 150)).astype(int)

	dosage_unit = rng.choice(
		["l_per_ha", "ml_per_m2", "gal_per_acre"],
		size=n_records,
		p=[0.8, 0.15, 0.05],
	)
	log_id = rng.integers(200000, 300000, size=n_records)

	logs = pd.DataFrame(
		{
			"log_id": log_id,
			"farm_id": farm_ids,
			"field_id": field_ids,
			"timestamp": timestamps,
			"crop": crops,
			"equipment": equipment,
			"operator_id": operator_id,
			"area_ha": area_ha,
			"dosage_value": dosage_value,
			"dosage_unit": dosage_unit,
			"spray_cost_cents": spray_cost_cents,
			"yield_kg": yield_kg.astype(int),
		}
	)

	# Seed duplicates (more common in struggling farm)
	struggling_indices = logs[logs["farm_id"] == "F-100"].index
	dup_count = max(10, n_records // 25)
	if len(struggling_indices) > 0:
		dup_idx = rng.choice(struggling_indices, size=min(dup_count // 2, len(struggling_indices)), replace=False)
		logs = pd.concat([logs, logs.loc[dup_idx]], ignore_index=True)

	# Inject missing dosage values (more in struggling farm)
	for farm_id, miss_rate in [("F-100", 0.08), ("F-200", 0.03), ("F-300", 0.01)]:
		farm_mask = logs["farm_id"] == farm_id
		missing_idx = logs[farm_mask].sample(frac=miss_rate, random_state=rng, replace=False).index
		logs.loc[missing_idx, "dosage_value"] = np.nan

	# Random casing drift (more common in struggling farm)
	for farm_id, drift_rate in [("F-100", 0.12), ("F-200", 0.05), ("F-300", 0.02)]:
		farm_mask = logs["farm_id"] == farm_id
		drift_idx = logs[farm_mask].sample(frac=drift_rate, random_state=rng, replace=False).index
		logs.loc[drift_idx, "equipment"] = logs.loc[drift_idx, "equipment"].str.upper()

	log_aliases = _choose_aliases(rng, LOG_ALIASES)
	logs = _apply_aliases(logs, log_aliases)

	# Weather data: make it more realistic with higher wind for struggling farms
	weather_time = pd.date_range(base_time, base_time + pd.Timedelta(days=90), freq="30min")
	weather_df_list = []
	
	for field_id in ["Field-A", "Field-B", "Field-C", "Field-D"]:
		field_weather_times = weather_time
		field_weather_records = len(field_weather_times)
		
		# Higher wind variability overall
		wind_speed_base = np.clip(rng.normal(5.2, 2.2, size=field_weather_records), 0.5, 25)
		
		weather_df_list.append(pd.DataFrame(
			{
				"field_id": field_id,
				"weather_time": field_weather_times,
				"wind_speed_mps": wind_speed_base,
				"temperature_c": rng.normal(16, 7, size=field_weather_records),
				"humidity": np.clip(rng.normal(0.58, 0.18, size=field_weather_records), 0.15, 0.99),
			}
		))
	
	weather = pd.concat(weather_df_list, ignore_index=True)

	# Separate ingest time to mimic delayed ingestion vs observation time.
	weather["ingest_time"] = weather["weather_time"] + pd.to_timedelta(
		rng.integers(0, 60 * 60 * 12, size=len(weather)), unit="s"
	)

	# Duplicate a slice with a later ingest_time to simulate late arrivals.
	late_idx = rng.choice(weather.index, size=30, replace=False)
	late_records = weather.loc[late_idx].copy()
	late_records["ingest_time"] += pd.Timedelta(hours=24)
	weather = pd.concat([weather, late_records], ignore_index=True)

	weather_aliases = _choose_aliases(rng, WEATHER_ALIASES)
	weather = _apply_aliases(weather, weather_aliases)

	financials = pd.DataFrame(
		{
			"crop": ["corn", "wheat", "soy", "barley"],
			"price_usd_per_kg": [0.28, 0.24, 0.35, 0.22],
			"baseline_yield_kg_per_ha": [8200, 7000, 3600, 6200],
			"cost_per_ha": [520, 480, 450, 470],
		}
	)

	fin_aliases = _choose_aliases(rng, FIN_ALIASES)
	financials = _apply_aliases(financials, fin_aliases)

	return {"spray_logs": logs, "weather": weather, "financials": financials}


def load_spray_logs(source_path: Path | str = DEFAULT_SPRAY_LOGS_CSV) -> pd.DataFrame:
	return pd.read_csv(source_path)


def standardize_columns(df: pd.DataFrame, alias_map: Dict[str, Iterable[str]]) -> pd.DataFrame:
	# Reverse alias map to canonical column names.
	reverse_map = {}
	for canonical, aliases in alias_map.items():
		for alias in aliases:
			reverse_map[alias] = canonical
	return df.rename(columns=reverse_map)


def convert_dosage_to_l_per_ha(values: pd.Series, units: pd.Series) -> pd.Series:
	factors = {
		"l_per_ha": 1.0,
		# 1 ml/m2 = 10 l/ha
		"ml_per_m2": 10.0,
		# 1 gal/acre -> l/ha
		"gal_per_acre": 3.78541 / 0.404686,
	}
	unit_factors = units.map(factors).fillna(1.0)
	return values.astype(float) * unit_factors


def remove_outliers_iqr(df: pd.DataFrame, columns: Iterable[str], k: float = 1.5) -> pd.DataFrame:
	filtered = df.copy()
	for col in columns:
		series = filtered[col].dropna()
		if series.empty:
			continue
		# IQR filter keeps values within k * IQR of the quartiles.
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		if iqr == 0:
			continue
		lower = q1 - k * iqr
		upper = q3 + k * iqr
		filtered = filtered[(filtered[col].isna()) | ((filtered[col] >= lower) & (filtered[col] <= upper))]
	return filtered


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
	optimized = df.copy()
	# Reduce memory for large frames before joins.
	# Explicitly include both 'object' and 'string' dtypes to be compatible
	# with pandas' upcoming string migration (silences Pandas4Warning).
	for col in optimized.select_dtypes(include=["object", "string"]).columns:
		optimized[col] = optimized[col].astype("category")
	for col in optimized.select_dtypes(include=["float64"]).columns:
		optimized[col] = pd.to_numeric(optimized[col], downcast="float")
	for col in optimized.select_dtypes(include=["int64"]).columns:
		optimized[col] = pd.to_numeric(optimized[col], downcast="integer")
	return optimized


def clean_spray_logs(df: pd.DataFrame) -> pd.DataFrame:
	cleaned = standardize_columns(df, LOG_ALIASES)
	cleaned = cleaned[list(CANONICAL_LOG_COLUMNS & set(cleaned.columns))].copy()

	cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
	cleaned["equipment"] = cleaned["equipment"].astype(str).str.strip().str.lower()
	cleaned["crop"] = cleaned["crop"].astype(str).str.strip().str.lower()

	cleaned["dosage_value"] = pd.to_numeric(cleaned["dosage_value"], errors="coerce")
	cleaned["area_ha"] = pd.to_numeric(cleaned["area_ha"], errors="coerce")
	cleaned["yield_kg"] = pd.to_numeric(cleaned["yield_kg"], errors="coerce")
	cleaned["spray_cost_cents"] = pd.to_numeric(cleaned["spray_cost_cents"], errors="coerce")

	# Normalize dosage into a single unit for analysis.
	cleaned["dosage_l_per_ha"] = convert_dosage_to_l_per_ha(
		cleaned["dosage_value"], cleaned["dosage_unit"]
	)
	cleaned["spray_cost_usd"] = cleaned["spray_cost_cents"] / 100.0
	cleaned["yield_kg_per_ha"] = cleaned["yield_kg"] / cleaned["area_ha"]

	# Keep one row per log_id after synthesis duplicates.
	cleaned = cleaned.drop_duplicates(subset=["log_id"])
	cleaned = remove_outliers_iqr(cleaned, ["dosage_l_per_ha", "yield_kg_per_ha"])

	cleaned = optimize_dtypes(cleaned)
	return cleaned


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
	cleaned = standardize_columns(df, WEATHER_ALIASES)
	cleaned = cleaned[list(CANONICAL_WEATHER_COLUMNS & set(cleaned.columns))].copy()

	cleaned["weather_time"] = pd.to_datetime(cleaned["weather_time"], errors="coerce")
	cleaned["ingest_time"] = pd.to_datetime(cleaned["ingest_time"], errors="coerce")

	for col in ["wind_speed_mps", "temperature_c", "humidity"]:
		cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

	cleaned = remove_outliers_iqr(cleaned, ["wind_speed_mps", "temperature_c"])
	cleaned = optimize_dtypes(cleaned)
	return cleaned


def clean_financials(df: pd.DataFrame) -> pd.DataFrame:
	cleaned = standardize_columns(df, FIN_ALIASES)
	cleaned = cleaned[list(CANONICAL_FIN_COLUMNS & set(cleaned.columns))].copy()
	cleaned["crop"] = cleaned["crop"].astype(str).str.strip().str.lower()
	for col in ["price_usd_per_kg", "baseline_yield_kg_per_ha", "cost_per_ha"]:
		cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
	cleaned = optimize_dtypes(cleaned)
	return cleaned


def logs_schema() -> DataFrameSchema:
	return DataFrameSchema(
		{
			"log_id": Column(int, nullable=False),
			"farm_id": Column(str, nullable=False),
			"field_id": Column(str, nullable=False),
			"timestamp": Column(pa.DateTime, nullable=False),
			"crop": Column(str, nullable=False),
			"equipment": Column(str, nullable=False),
			"operator_id": Column(int, nullable=False),
			"area_ha": Column(float, Check.in_range(0.5, 200), nullable=False),
			"dosage_l_per_ha": Column(float, Check.in_range(0.1, 25), nullable=True),
			"spray_cost_usd": Column(float, Check.ge(0), nullable=True),
			"yield_kg": Column(float, Check.ge(0), nullable=True),
			"yield_kg_per_ha": Column(float, Check.in_range(100, 15000), nullable=True),
		},
		strict=False,
		coerce=True,
		# Enforce a null-rate threshold for dosage after cleaning.
		checks=[Check(lambda df: df["dosage_l_per_ha"].isna().mean() < 0.1)],
	)


def weather_schema() -> DataFrameSchema:
	return DataFrameSchema(
		{
			"field_id": Column(str, nullable=False),
			"weather_time": Column(pa.DateTime, nullable=False),
			"ingest_time": Column(pa.DateTime, nullable=True),
			"wind_speed_mps": Column(float, Check.in_range(0, 30), nullable=True),
			"temperature_c": Column(float, Check.in_range(-15, 45), nullable=True),
			"humidity": Column(float, Check.in_range(0.1, 1.0), nullable=True),
		},
		strict=False,
		coerce=True,
	)


def financials_schema() -> DataFrameSchema:
	return DataFrameSchema(
		{
			"crop": Column(str, nullable=False),
			"price_usd_per_kg": Column(float, Check.in_range(0.05, 5.0), nullable=False),
			"baseline_yield_kg_per_ha": Column(float, Check.in_range(500, 15000), nullable=False),
			"cost_per_ha": Column(float, Check.in_range(10, 3000), nullable=False),
		},
		strict=False,
		coerce=True,
	)


def validate_data(logs: pd.DataFrame, weather: pd.DataFrame, financials: pd.DataFrame) -> None:
	logs_schema().validate(logs, lazy=True)
	weather_schema().validate(weather, lazy=True)
	financials_schema().validate(financials, lazy=True)


def merge_data(
	logs: pd.DataFrame,
	weather: pd.DataFrame,
	financials: pd.DataFrame,
	weather_tolerance: str = "2h",
) -> pd.DataFrame:
	# Merge per field to keep merge_asof's ordering rules simple and reliable.
	logs_sorted = logs.dropna(subset=["timestamp"]).sort_values(["field_id", "timestamp"]).reset_index(drop=True)
	weather_sorted = weather.dropna(subset=["weather_time"]).sort_values(["field_id", "weather_time"]).reset_index(drop=True)

	merged_parts = []
	for field_id, log_group in logs_sorted.groupby("field_id", sort=False):
		weather_group = weather_sorted[weather_sorted["field_id"] == field_id]
		if weather_group.empty:
			matched = log_group.copy()
			for col in ["weather_time", "ingest_time", "wind_speed_mps", "temperature_c", "humidity"]:
				if col not in matched.columns:
					matched[col] = pd.NA
		else:
			matched = pd.merge_asof(
				log_group.sort_values("timestamp"),
				weather_group.sort_values("weather_time"),
				left_on="timestamp",
				right_on="weather_time",
				tolerance=pd.Timedelta(weather_tolerance),
				direction="backward",
			)
		merged_parts.append(matched)

	merged = pd.concat(merged_parts, ignore_index=True)

	merged = merged.merge(financials, on="crop", how="left")
	return merged


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
	enriched = df.copy()
	revenue_per_ha = enriched["yield_kg_per_ha"] * enriched["price_usd_per_kg"]
	spray_cost_per_ha = enriched["spray_cost_usd"] / enriched["area_ha"]
	enriched["cost_per_spray"] = enriched["spray_cost_usd"]
	enriched["yield_delta"] = enriched["yield_kg_per_ha"] - enriched["baseline_yield_kg_per_ha"]
	# Profit margin normalized by revenue per ha.
	enriched["profit_margin"] = (revenue_per_ha - spray_cost_per_ha - enriched["cost_per_ha"]) / revenue_per_ha
	# Proxy for wasted spray due to wind.
	enriched["wind_waste_proxy"] = enriched["wind_speed_mps"] * enriched["dosage_l_per_ha"]
	return enriched


def compute_correlations(df: pd.DataFrame) -> pd.Series:
	cols = ["wind_speed_mps", "wind_waste_proxy", "dosage_l_per_ha", "profit_margin"]
	return df[cols].corr().stack().sort_values(ascending=False)


def create_plots(df: pd.DataFrame) -> None:
	import seaborn as sns
	import matplotlib.pyplot as plt

	sns.regplot(data=df, x="wind_speed_mps", y="wind_waste_proxy")
	plt.title("Wind Speed vs Waste Proxy")
	plt.show()

	sns.regplot(data=df, x="dosage_l_per_ha", y="profit_margin")
	plt.title("Dosage Precision vs Profit")
	plt.show()

	sns.boxplot(data=df, x="crop", y="profit_margin")
	plt.title("ROI by Crop")
	plt.show()


def scalability_recommendation(rows: int, daily_ingestion: bool) -> str:
	if rows >= 10_000_000 or daily_ingestion:
		return "Use Polars or PySpark for compute, and BigQuery/Snowflake + dbt for storage."
	return "Pandas is sufficient; consider Polars for lazy optimization when nearing 10M rows."


def orchestration_plan() -> Dict[str, str]:
	return {
		"prefect": "Rapid deployment and Python-first workflows.",
		"airflow": "Enterprise DAGs with strong scheduler ecosystem.",
		"dagster": "Asset-centric orchestration with lineage and checks.",
		"monitoring": "Alert on schema drift using pandera checks and run logs.",
	}


def save_pipeline_outputs(
	logs: pd.DataFrame,
	weather: pd.DataFrame,
	financials: pd.DataFrame,
	metrics: pd.DataFrame,
	output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Path]:
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	paths = {
		"cleaned_spray_logs": output_path / "cleaned_spray_logs.csv",
		"cleaned_weather": output_path / "cleaned_weather.csv",
		"cleaned_financials": output_path / "cleaned_financials.csv",
		"final_metrics": output_path / "final_metrics.csv",
	}

	logs.to_csv(paths["cleaned_spray_logs"], index=False)
	weather.to_csv(paths["cleaned_weather"], index=False)
	financials.to_csv(paths["cleaned_financials"], index=False)
	metrics.to_csv(paths["final_metrics"], index=False)
	return paths


def run_pipeline(
	n_records: int = 800,
	seed: int = 42,
	spray_logs_path: Path | str = DEFAULT_SPRAY_LOGS_CSV,
	output_dir: Path | str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	data = synthesize_data(n_records=n_records, seed=seed)
	logs_source = load_spray_logs(spray_logs_path) if Path(spray_logs_path).exists() else data["spray_logs"]
	logs = clean_spray_logs(logs_source)
	weather = clean_weather(data["weather"])
	financials = clean_financials(data["financials"])
	validate_data(logs, weather, financials)

	merged = merge_data(logs, weather, financials)
	metrics = compute_metrics(merged)
	if output_dir is not None:
		save_pipeline_outputs(logs, weather, financials, metrics, output_dir=output_dir)
	return logs, weather, metrics


if __name__ == "__main__":
	_, _, metrics_df = run_pipeline(output_dir=DEFAULT_OUTPUT_DIR)
	print(metrics_df.head())

