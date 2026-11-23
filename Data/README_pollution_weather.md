# Morocco city pollution and weather dataset

`morocco_city_pollution_weather_hourly.csv` contains hourly rows for Casablanca,
Marrakech, Fes, and Tangier. It is assembled from the Open-Meteo Air Quality API
(CAMS Global modeled concentrations) and Open-Meteo Historical Weather API
(ERA5 reanalysis). It is suitable for experimenting with prediction workflows,
but the pollutant columns are model estimates, not measurements from local
monitoring stations. Validate against station observations before making
real-world claims.

The downloader requests the most recent 92 days because the air-quality API's
historical window is limited to 92 days. Weather archive data can lag by several
days; weather fields for hours not yet available from ERA5 are blank. The CSV
includes pollutant concentrations in µg/m³, standard weather variables, city
coordinates, and basic calendar features. Missing values should be handled in
the modeling pipeline.

To refresh the CSV, run from this folder or any directory:

```powershell
python ..\Scripts\download_city_pollution_weather.py
```

Sources and documentation:

- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- Air quality attribution: Copernicus Atmosphere Monitoring Service (CAMS).
- Weather source: ERA5 reanalysis. Open-Meteo data is provided under CC BY 4.0;
  retain attribution when redistributing derived files.
