import ee
import pandas as pd
import datetime

def fetch_satellite_data():
    print("Authenticating with Google Earth Engine...")
    try:
        ee.Initialize(project='sugarcane-ndvi-project')
    except Exception as e:
        print("Earth Engine not authenticated. Triggering login...")
        ee.Authenticate()
        ee.Initialize(project='sugarcane-ndvi-project')

    print("Connecting to Copernicus Sentinel-2 Archive...")
    
    # We will sample a few key Sugarcane districts in Punjab based on your master dataset
    # You can expand this list later!
    locations = {
        'Faisalabad': [31.4187, 73.0791],
        'Sargodha': [32.0836, 72.6711],
        'Bahawalpur': [29.3956, 71.6836]
    }

    start_date = '2020-05-01'
    end_date = '2025-01-01'
    
    all_data = []

    for district, coords in locations.items():
        print(f"Extracting historical NDVI for {district}...")
        poi = ee.Geometry.Point([coords[1], coords[0]]) # Longitude, Latitude

        # Load Sentinel-2 Surface Reflectance Data (Cloud Filtered)
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(poi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

        def calculate_ndvi(image):
            # NDVI = (NIR - RED) / (NIR + RED)
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            mean_ndvi = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=poi,
                scale=10
            ).get('NDVI')
            date = image.date().format('YYYY-MM-dd')
            return ee.Feature(None, {'Date': date, 'NDVI': mean_ndvi, 'District': district})

        # Apply function and fetch results
        ndvi_features = s2.map(calculate_ndvi).getInfo()['features']

        for feature in ndvi_features:
            props = feature['properties']
            if props.get('NDVI') is not None:
                all_data.append({
                    'District': props['District'],
                    'Date': props['Date'],
                    'Calculated_NDVI': round(props['NDVI'], 3)
                })

    df_satellite = pd.DataFrame(all_data)
    df_satellite['Date'] = pd.to_datetime(df_satellite['Date'])
    df_satellite = df_satellite.sort_values(['District', 'Date']).reset_index(drop=True)
    
    print("\n--- Satellite Extraction Complete! ---")
    print(df_satellite.head())
    
    df_satellite.to_csv('sentinel2_ndvi_5years.csv', index=False)
    print("\nSuccessfully saved as: sentinel2_ndvi_5years.csv")

if __name__ == "__main__":
    fetch_satellite_data()