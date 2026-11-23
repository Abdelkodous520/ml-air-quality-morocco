"""Download recent hourly weather and modeled air quality for four Moroccan cities.

Source: Open-Meteo Air Quality API (CAMS) and Historical Weather API (ERA5).
Run from any directory with Python 3.10+: python download_city_pollution_weather.py
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "morocco_city_pollution_weather_hourly.csv"
PAST_DAYS = 92
WEATHER_DELAY_DAYS = 5  # ERA5 archive is not immediate; recent hours may lack weather.

CITIES = {
    "Casablanca": (33.5731, -7.5898),
    "Marrakech": (31.6295, -7.9811),
    "Fes": (34.0331, -5.0003),
    "Tangier": (35.7595, -5.8340),
}

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]
POLLUTION_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide"]


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "MoroccoPollutionWeatherDataset/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "API returned an error"))
    return payload


def url_for(base: str, params: dict) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"


def main() -> None:
    today = date.today()
    aq_start = today - timedelta(days=PAST_DAYS)
    aq_end = today - timedelta(days=1)
    weather_end = min(aq_end, today - timedelta(days=WEATHER_DELAY_DAYS))
    weather_start = aq_start
    rows: list[dict] = []

    for city, (lat, lon) in CITIES.items():
        aq = get_json(url_for("https://air-quality-api.open-meteo.com/v1/air-quality", {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(POLLUTION_VARS),
            "past_days": PAST_DAYS, "forecast_days": 0,
            "timezone": "Africa/Casablanca", "domains": "cams_global",
        }))
        weather = get_json(url_for("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": lat, "longitude": lon,
            "start_date": weather_start.isoformat(), "end_date": weather_end.isoformat(),
            "hourly": ",".join(WEATHER_VARS), "timezone": "Africa/Casablanca",
        }))

        aq_hourly = aq["hourly"]
        weather_hourly = weather["hourly"]
        weather_by_time = {
            timestamp: {key: values[i] for key, values in weather_hourly.items() if key != "time"}
            for i, timestamp in enumerate(weather_hourly["time"])
        }
        for i, timestamp in enumerate(aq_hourly["time"]):
            try:
                dt = datetime.fromisoformat(timestamp)
            except ValueError:
                dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M")
            weather_values = weather_by_time.get(timestamp, {})
            row = {
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "datetime_local": timestamp,
                "date": dt.date().isoformat(),
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "month": dt.month,
                "pm2_5_ug_m3": aq_hourly["pm2_5"][i],
                "pm10_ug_m3": aq_hourly["pm10"][i],
                "no2_ug_m3": aq_hourly["nitrogen_dioxide"][i],
                "ozone_ug_m3": aq_hourly["ozone"][i],
                "so2_ug_m3": aq_hourly["sulphur_dioxide"][i],
            }
            row.update(weather_values)
            rows.append(row)

    if not rows:
        raise RuntimeError("No hourly rows returned by the APIs")
    columns = list(rows[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows):,} rows to {OUTPUT}")
    print(f"Pollution window: {aq_start} through {aq_end} (local hourly timestamps)")
    print(f"Weather window: {weather_start} through {weather_end}; recent hours may be blank")


if __name__ == "__main__":
    main()
