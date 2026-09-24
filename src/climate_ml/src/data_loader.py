import os
import time
import logging
from typing import List, Any, Dict, Union
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
import numpy as np
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CairoGridExtractor:
    """Responsible for generating geographic grid cells and extracting OSM spatial features for regions."""

    def __init__(self, place_names: List[str], cell_size: float = 0.01):
        self.place_names = place_names
        self.cell_size = cell_size

    def _generate_grid_for_polygon(self, polygon, region_name: str) -> gpd.GeoDataFrame:
        xmin, ymin, xmax, ymax = polygon.bounds
        grid_cells = []
        cx = xmin
        while cx < xmax:
            cy = ymin
            while cy < ymax:
                cell = box(cx, cy, cx + self.cell_size, cy + self.cell_size)
                if polygon.intersects(cell):
                    grid_cells.append(cell)
                cy += self.cell_size
            cx += self.cell_size

        if not grid_cells:
            return gpd.GeoDataFrame()

        district_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")
        district_gdf['district'] = region_name
        district_gdf['grid_id'] = [f"{region_name}_{i}" for i in range(len(district_gdf))]
        district_gdf['centroid'] = district_gdf.geometry.centroid
        district_gdf['lat'] = district_gdf['centroid'].y
        district_gdf['lon'] = district_gdf['centroid'].x
        return district_gdf

    def _fetch_osm_features(self, polygon):
        """Fetches various OpenStreetMap spatial layers safely."""
        features = {}
        tags_config: Dict[str, Dict[str, Union[bool, List[str], str]]] = {
            'greenery': {"leisure": ["park", "garden"], "landuse": ["grass", "recreation_ground", "forest"]},
            'building': {"building": True},
            'road': {"highway": True},
            'water': {"natural": ["water"], "waterway": True},
            'poi': {"amenity": True, "shop": True, "office": True},
            'bare': {"natural": ["sand", "bare_rock", "scrub", "desert", "earth"], "landuse": ["brownfield", "construction", "garages", "depot"]}
        }
        
        for key, tags in tags_config.items():
            try:
                features[key] = ox.features_from_polygon(polygon, tags=tags)
            except Exception:
                features[key] = None
        return features

    def extract_spatial_features(self) -> pd.DataFrame:
        all_grids = []

        for place in self.place_names:
            region_name = place.split(',')[0].replace(" Governorate", "")
            logger.info(f"Processing Region: {region_name}")

            try:
                gdf_place = ox.geocode_to_gdf(place)
                polygon = gdf_place.geometry.values[0]
                
                district_gdf = self._generate_grid_for_polygon(polygon, region_name)
                if district_gdf.empty:
                    continue

                district_area = district_gdf.geometry.area.iloc[0]
                osm_feats = self._fetch_osm_features(polygon)

                # Compute spatial metrics per cell
                records = []
                for _, row in district_gdf.iterrows():
                    cell: Any = row['geometry']
                    centroid = cell.centroid

                    # 1. Greenery
                    g_feat = osm_feats['greenery']
                    green_area = float(g_feat.geometry.intersection(cell).area.sum()) if g_feat is not None and not g_feat.empty else 0.0

                    # 2. Buildings & Levels
                    b_feat = osm_feats['building']
                    bld_area = float(b_feat.geometry.intersection(cell).area.sum()) if b_feat is not None and not b_feat.empty else 0.0

                    avg_lvl = 1.0
                    if b_feat is not None and not b_feat.empty:
                        intersected_bldgs = b_feat[b_feat.geometry.intersects(cell)]
                        if not intersected_bldgs.empty and 'building:levels' in intersected_bldgs.columns:
                            lvls_list = []
                            for val in intersected_bldgs['building:levels']:
                                try:
                                    lvls_list.append(float(val))
                                except (ValueError, TypeError):
                                    pass
                            if lvls_list:
                                avg_lvl = float(sum(lvls_list) / len(lvls_list))

                    # 3. Roads
                    r_feat = osm_feats['road']
                    road_len = float(r_feat.geometry.intersection(cell).length.sum()) if r_feat is not None and not r_feat.empty else 0.0

                    # 4. Water Distance
                    w_feat = osm_feats['water']
                    min_dist = 0.0
                    if w_feat is not None and not w_feat.empty:
                        min_dist = min([centroid.distance(geom) for geom in w_feat.geometry], default=0.0)

                    # 5. POI Density
                    p_feat = osm_feats['poi']
                    poi_count = len(p_feat[p_feat.geometry.intersects(cell)]) if p_feat is not None and not p_feat.empty else 0

                    # 6. Bare Ground (vectorized & fixes undefined b_geom)
                    ba_feat = osm_feats['bare']
                    bare_area = float(ba_feat.geometry.intersection(cell).area.sum()) if ba_feat is not None and not ba_feat.empty else 0.0

                    records.append({
                        'greenery_area': green_area,
                        'greenery_density': green_area / district_area,
                        'building_area': bld_area,
                        'building_density': bld_area / district_area,
                        'avg_building_levels': avg_lvl,
                        'road_density': road_len / district_area,
                        'distance_to_water': float(min_dist),
                        'poi_density': float(poi_count) / district_area,
                        'bare_ground_density': bare_area / district_area,
                    })

                metrics_df = pd.DataFrame(records, index=district_gdf.index)
                district_df = pd.concat([district_gdf.drop(columns=['geometry', 'centroid'], errors='ignore'), metrics_df], axis=1)
                all_grids.append(pd.DataFrame(district_df))

            except Exception as e:
                logger.error(f"Error in region {place}: {e}")

        return pd.concat(all_grids, ignore_index=True) if all_grids else pd.DataFrame()


class EnvironmentalFeatureFetcher:
    """Responsible for fetching external APIs and simulating environmental layers."""

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

    def fetch_temperatures(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Fetching Open-Meteo Temperature data...")
        temperatures = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={row['lat']}&longitude={row['lon']}&start_date={self.start_date}&end_date={self.end_date}&daily=temperature_2m_mean"
            try:
                res = requests.get(url, timeout=5)
                if res.status_code == 200:
                    temps = [t for t in res.json().get('daily', {}).get('temperature_2m_mean', []) if t is not None]
                    mean_temp = sum(temps) / len(temps) if temps else 25.0
                else:
                    mean_temp = 25.0
            except Exception:
                mean_temp = 25.0
            temperatures.append(mean_temp)
            time.sleep(0.02)

        df['mean_temperature'] = temperatures
        return df

    def add_simulated_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ndvi_mean'] = df.apply(lambda r: max(0.0, 0.25 - (r['building_density'] * 0.1)), axis=1)
        df['elevation'] = df.apply(lambda r: 40.0 + (r['lat'] - 30.0) * 500.0, axis=1)
        df['nighttime_lights_intensity'] = df.apply(lambda r: max(2.0, 15.0 + (r['road_density'] * 0.002)), axis=1)
        df['green_building_ratio'] = df['greenery_area'] / (df['building_area'] + 1e-6)
        return df


class DataIngestionPipeline:
    """Orchestrates the entire data loading, processing, and saving pipeline."""

    def __init__(self, place_names: List[str], output_path: str):
        self.place_names = place_names
        self.output_path = output_path
        self.extractor = CairoGridExtractor(place_names)
        self.env_fetcher = EnvironmentalFeatureFetcher("2025-01-01", "2025-12-31")

    def run(self) -> pd.DataFrame:
        logger.info("Starting Unified Greater Cairo Data Ingestion Pipeline...")
        
        # 1. Extract spatial & OSM features
        master_df = self.extractor.extract_spatial_features()
        if master_df.empty:
            logger.warning("No data extracted!")
            return master_df

        # 2. Fetch environmental features & simulate metrics
        master_df = self.env_fetcher.fetch_temperatures(master_df)
        master_df = self.env_fetcher.add_simulated_metrics(master_df)

        # 3. Save to CSV
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        master_df.to_csv(self.output_path, index=False)
        logger.info(f"Pipeline Complete! Dataset saved to '{self.output_path}' with shape: {master_df.shape}")
        
        return master_df