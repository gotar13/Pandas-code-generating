import math

import pandas as pd

from main import (
    clean_financials,
    clean_spray_logs,
    clean_weather,
    convert_dosage_to_l_per_ha,
    remove_outliers_iqr,
    synthesize_data,
    validate_data,
)


def test_schema_validation():
    data = synthesize_data(n_records=200, seed=7)
    logs = clean_spray_logs(data["spray_logs"])
    weather = clean_weather(data["weather"])
    financials = clean_financials(data["financials"])
    validate_data(logs, weather, financials)


def test_outliers_removed():
    df = pd.DataFrame(
        {
            "dosage_l_per_ha": [1.0, 1.1, 1.2, 20.0],
            "yield_kg_per_ha": [3000, 3100, 3200, 50000],
        }
    )
    filtered = remove_outliers_iqr(df, ["dosage_l_per_ha", "yield_kg_per_ha"])
    assert len(filtered) == 3


def test_unit_conversions():
    values = pd.Series([1.0, 1.0, 1.0])
    units = pd.Series(["l_per_ha", "ml_per_m2", "gal_per_acre"])
    converted = convert_dosage_to_l_per_ha(values, units)
    assert math.isclose(converted.iloc[0], 1.0, rel_tol=1e-6)
    assert math.isclose(converted.iloc[1], 10.0, rel_tol=1e-6)
    assert math.isclose(converted.iloc[2], 3.78541 / 0.404686, rel_tol=1e-6)
