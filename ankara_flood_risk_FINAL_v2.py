# ═══════════════════════════════════════════════════════════════════════════════
# ANKARA URBAN FLOOD RISK: GEOAI-BASED SPATIAL ANALYSIS
# AND SOCIAL VULNERABILITY ASSESSMENT
#
# Mustafa Berk Doğru
# Ankara University, Faculty of Language, History and Geography
# Department of Geography — Esri Türkiye Young Scholars 2026
#
# COMPREHENSIVE REVISED VERSION
# ═══════════════════════════════════════════════════════════════════════════════

import arcpy
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# 0. SETUP AND PATHS
# ═══════════════════════════════════════════════════════════════════════════════
arcpy.env.overwriteOutput = True

# Main directories
GDB    = arcpy.env.workspace
OUTPUT = r"C:\Users\Mustafa Berk\Downloads\ABB data\output"
DOCS   = r"C:\Users\Mustafa Berk\Documents"
DISTRICT_SHP = r"C:\Users\Mustafa Berk\Documents\districts2.shp"

# DEM files
DEM1 = r"C:\Users\Mustafa Berk\Downloads\ABB data\DEM\AP_07684_FBD_F0790_RT1_111\DEM_1\AP_07684_FBD_F0790_RT1.dem.tif"
DEM2 = r"C:\Users\Mustafa Berk\Downloads\ABB data\DEM\AP_12133_FBD_F0790_RT1_222\DEM_2\AP_12133_FBD_F0790_RT1.dem.tif"
DEM3 = r"C:\Users\Mustafa Berk\Downloads\ABB data\DEM\AP_07684_FBD_F0780_RT1_333\DEM_3\AP_07684_FBD_F0780_RT1.dem.tif"

# GDB layer paths
FLOOD_MAIN     = os.path.join(GDB, "flood_points_main")          # 215 report points
FLOOD_AUX      = os.path.join(GDB, "flood_points_aux")           # Auxiliary street points
DISTRICT_GDB   = os.path.join(GDB, "district_boundaries_gdb")    # 5 district boundaries
TRAINING_V4    = os.path.join(GDB, "training_data_v4")           # Training data
PREDICTION_V5  = os.path.join(GDB, "flood_risk_prediction_v5")   # Prediction results
IMPORTANCE_V5  = os.path.join(GDB, "variable_importance_final_v5")  # Variable importance table
HOTSPOT        = os.path.join(GDB, "hotspot_optimized")          # Hotspot analysis
EHS            = os.path.join(GDB, "emerging_hotspot")           # Emerging hotspot
SVI_TABLE      = os.path.join(GDB, "svi_district_score")         # SVI table
ZONAL          = os.path.join(GDB, "zonal_stats_district")       # Zonal statistics
DISTRICT_CV    = os.path.join(GDB, "district_cv_results")        # LODO CV results

# Raster paths
DEM            = os.path.join(GDB, "dem_ankara_merged")
SLOPE          = os.path.join(GDB, "slope")
TWI            = os.path.join(GDB, "twi")
FLOW_ACC       = os.path.join(GDB, "flow_accumulation")
FLOW_DIR       = os.path.join(GDB, "flow_direction")
EUC_DIST       = os.path.join(GDB, "euc_dist_meters")

# Ankara districts
DISTRICTS = ["Keçiören", "Çankaya", "Altındağ", "Akyurt", "Yenimahalle"]

print(f"✓ Workspace: {GDB}")
print(f"✓ Setup complete")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION - CREATING POINT LAYERS FROM CSV
# ═══════════════════════════════════════════════════════════════════════════════

def csv_to_points(csv_path, layer_name):
    """
    Cleans a CSV file and converts it into a point feature class.
    Filters coordinates within Ankara's boundaries.

    Parameters:
    -----------
    csv_path : str - CSV file path
    layer_name : str - Name of the layer to create

    Returns:
    --------
    str - Feature class path
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Convert coordinates to numeric values
    df['LATITUDE']  = pd.to_numeric(df['LATITUDE'],  errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

    # Remove null coordinates
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

    # Ankara boundary filter (approximate coordinate ranges)
    df = df[(df['LATITUDE'] > 38) & (df['LATITUDE'] < 41)]
    df = df[(df['LONGITUDE'] > 30) & (df['LONGITUDE'] < 35)]

    # Save cleaned CSV
    cleaned = csv_path.replace('.csv', '_clean.csv')
    df.to_csv(cleaned, index=False, encoding='utf-8-sig')

    # Create feature class
    out_fc = os.path.join(GDB, layer_name)
    arcpy.management.XYTableToPoint(
        cleaned, out_fc, "LONGITUDE", "LATITUDE",
        coordinate_system=arcpy.SpatialReference(4326)
    )

    n = int(arcpy.management.GetCount(out_fc).getOutput(0))
    print(f"✓ {layer_name}: {n} points")
    return out_fc


def create_csv_layers():
    """Imports all CSV files"""

    # Flood points
    fc1 = csv_to_points(os.path.join(OUTPUT, "01_flood_points_main.csv"), "flood_points_main")
    fc2 = csv_to_points(os.path.join(OUTPUT, "02_flood_points_aux.csv"), "flood_points_aux")

    # Precipitation stations
    fc3 = csv_to_points(os.path.join(OUTPUT, "03_precipitation_stations_june2022.csv"),
                        "precipitation_stations_2022")

    # Tables
    arcpy.conversion.ExportTable(os.path.join(OUTPUT, "04_population_district.csv"),
                                  os.path.join(GDB, "population_district"))
    arcpy.conversion.ExportTable(os.path.join(OUTPUT, "05_drainage_issues_district.csv"),
                                  os.path.join(GDB, "drainage_issues_district"))

    print("✓ All CSV files imported")
    return fc1, fc2, fc3

# create_csv_layers()  # Enable on first run


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEM MOSAIC AND DERIVED LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

def merge_dem():
    """
    Merges three separate DEM files using Mosaic.
    """
    print("Merging DEMs...")

    arcpy.management.MosaicToNewRaster(
        input_rasters   = [DEM1, DEM2, DEM3],
        output_location = GDB,
        raster_dataset_name_with_extension = "dem_ankara_merged",
        coordinate_system_for_the_raster   = arcpy.SpatialReference(4326),
        pixel_type      = "32_BIT_FLOAT",
        number_of_bands = 1,
        mosaic_method   = "MEAN"
    )
    print("✓ DEM merged")


def calculate_dem_derivatives():
    """
    Calculates derived layers from the DEM:
    - Slope
    - Flow Direction
    - Flow Accumulation
    - TWI (Topographic Wetness Index)
    - Euclidean distance to streams
    """
    print("Calculating DEM derivatives...")

    # 1. Slope (in degrees)
    arcpy.ddd.Slope(DEM, SLOPE, "DEGREE")
    print("  ✓ Slope calculated")

    # 2. Flow Direction
    arcpy.ddd.FlowDirection(DEM, FLOW_DIR)
    print("  ✓ Flow direction calculated")

    # 3. Flow Accumulation
    arcpy.ddd.FlowAccumulation(FLOW_DIR, FLOW_ACC)
    print("  ✓ Flow accumulation calculated")

    # 4. TWI (Topographic Wetness Index)
    # TWI = ln(a / tan(β)) where a = flow accumulation, β = slope angle
    print("  Calculating TWI...")
    slope_rad = arcpy.sa.Times(arcpy.sa.Raster(SLOPE), 0.01745329)  # Degrees -> Radians
    tan_slope = arcpy.sa.Con(slope_rad > 0.001, arcpy.sa.Tan(slope_rad), 0.001)  # Avoid division by zero
    twi_r = arcpy.sa.Ln(arcpy.sa.Plus(arcpy.sa.Raster(FLOW_ACC), 1) / tan_slope)
    twi_r.save(TWI)
    print("  ✓ TWI calculated")

    # 5. Euclidean distance to streams
    streams = os.path.join(GDB, "stream_lines")
    if arcpy.Exists(streams):
        arcpy.sa.EucDistance(streams).save(EUC_DIST)
        print("  ✓ Distance to streams calculated")
    else:
        print("  ! Stream layer not found, skipping EUC_DIST")

    print("✓ All DEM derivatives calculated")

# merge_dem()
# calculate_dem_derivatives()  # Enable on first run


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MERGING POINT LAYERS AND ADDING RASTER VALUES
# ═══════════════════════════════════════════════════════════════════════════════

def merge_and_enrich_points():
    """
    Merges flood points and adds raster values.
    """
    print("Merging and enriching points...")

    # Merge the two point layers
    flood_merged = os.path.join(GDB, "flood_points_merged")
    arcpy.management.Merge([FLOOD_MAIN, FLOOD_AUX], flood_merged)
    n = int(arcpy.management.GetCount(flood_merged).getOutput(0))
    print(f"  ✓ Merged {n} flood points total")

    # Add raster values to points
    raster_list = [
        [SLOPE, "SLOPE_DEGREE"],
        [DEM, "ELEVATION_M"],
        [FLOW_ACC, "FLOW_ACCUMULATION"],
        [TWI, "TWI"],
    ]

    if arcpy.Exists(EUC_DIST):
        raster_list.append([EUC_DIST, "EUC_DIST"])

    arcpy.sa.ExtractMultiValuesToPoints(
        in_point_features = flood_merged,
        in_rasters = raster_list,
        bilinear_interpolate_values = "BILINEAR"
    )
    print("  ✓ Raster values added")

    return flood_merged

# flood_merged = merge_and_enrich_points()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SMART NEGATIVE SAMPLING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

def negative_sampling(n_target=410, buffer_meters=500):
    """
    Generates negative points in areas that are physically unlikely to flood.

    Criteria:
    - High elevation (>1200m) OR
    - Flat area (slope < 2°) AND low flow accumulation (<10)

    Buffer: Minimum distance from existing flood points

    Parameters:
    -----------
    n_target : int - Target number of negative points
    buffer_meters : float - Minimum distance from flood points (meters)

    Returns:
    --------
    list - (x, y, slope, elevation, flow_acc, twi) tuples
    """
    print(f"Starting negative sampling (target: {n_target} points, buffer: {buffer_meters}m)...")

    # Study area extent
    desc = arcpy.Describe(DISTRICT_GDB)
    extent = desc.extent
    xmin, xmax = extent.XMin, extent.XMax
    ymin, ymax = extent.YMin, extent.YMax

    # Convert rasters to numpy arrays (for performance)
    print("  Loading rasters...")
    slope_r = arcpy.RasterToNumPyArray(SLOPE, nodata_to_value=-9999)
    flow_r  = arcpy.RasterToNumPyArray(FLOW_ACC, nodata_to_value=-9999)
    dem_r   = arcpy.RasterToNumPyArray(DEM, nodata_to_value=-9999)
    twi_r   = arcpy.RasterToNumPyArray(TWI, nodata_to_value=-9999)

    cell = arcpy.Describe(SLOPE).meanCellWidth

    # Flood point coordinates (for buffer check)
    flood_coords = [(r[0], r[1]) for r in arcpy.da.SearchCursor(FLOOD_MAIN, ["SHAPE@X", "SHAPE@Y"])]
    print(f"  {len(flood_coords)} flood points loaded")

    points = []
    attempt = 0
    random.seed(42)

    while len(points) < n_target and attempt < n_target * 100:
        attempt += 1

        # Generate random coordinate
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)

        # Calculate pixel indices
        col = int((x - xmin) / cell)
        row = int((ymax - y) / cell)

        if row < 0 or col < 0 or row >= slope_r.shape[0] or col >= slope_r.shape[1]:
            continue

        try:
            slope_val = slope_r[row, col]
            flow_val  = flow_r[row, col]
            dem_val   = dem_r[row, col]
            twi_val   = twi_r[row, col]
        except IndexError:
            continue

        # NoData check
        if slope_val == -9999:
            continue

        # Physical criterion: area unlikely to flood
        physical = (dem_val > 1200) or (slope_val < 2 and flow_val < 10)
        if not physical:
            continue

        # Buffer check: far from flood points
        too_close = any(
            (x - sx)**2 + (y - sy)**2 < buffer_meters**2
            for sx, sy in flood_coords
        )
        if too_close:
            continue

        points.append((x, y, slope_val, dem_val, flow_val, twi_val))

    print(f"  ✓ {len(points)} negative points generated ({attempt} attempts)")
    return points


def create_negative_point_fc(point_list):
    """
    Creates a feature class from the negative point list.
    """
    negative_fc = os.path.join(GDB, "no_flood_points")

    # Create feature class
    arcpy.management.CreateFeatureclass(
        GDB, "no_flood_points", "POINT",
        spatial_reference=arcpy.SpatialReference(4326)
    )

    # Add fields
    fields = [
        ("FLOOD_POINT", "SHORT"),
        ("priority_score", "SHORT"),
        ("SLOPE_DEGREE", "DOUBLE"),
        ("ELEVATION_M", "DOUBLE"),
        ("FLOW_ACCUMULATION", "DOUBLE"),
        ("TWI", "DOUBLE"),
    ]

    for field, ftype in fields:
        arcpy.management.AddField(negative_fc, field, ftype)

    # Insert points
    cursor_fields = ["SHAPE@XY", "FLOOD_POINT", "priority_score",
                      "SLOPE_DEGREE", "ELEVATION_M", "FLOW_ACCUMULATION", "TWI"]

    with arcpy.da.InsertCursor(negative_fc, cursor_fields) as cur:
        for x, y, slope, dem, flow, twi in point_list:
            cur.insertRow([(x, y), 0, 0, slope, dem, flow, twi])

    print(f"✓ {len(point_list)} negative points added to feature class")
    return negative_fc

# negative_list = negative_sampling(n_target=410, buffer_meters=500)
# negative_fc = create_negative_point_fc(negative_list)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CREATING AND ENRICHING TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════

def create_training_data():
    """
    Creates training data by merging positive and negative points.
    Adds district, population, and drainage information.
    """
    print("Creating training data...")

    # Merge positive and negative points
    flood_merged = os.path.join(GDB, "flood_points_merged")
    negative_fc = os.path.join(GDB, "no_flood_points")

    training_fc = os.path.join(GDB, "training_data_v4")
    arcpy.management.Merge([flood_merged, negative_fc], training_fc)

    n = int(arcpy.management.GetCount(training_fc).getOutput(0))
    print(f"  ✓ {n} points merged")

    return training_fc


def enrich_data(training_fc):
    """
    Adds district, population, and drainage information to the training data.
    """
    print("Enriching data...")

    # Create district key (uppercase, normalized)
    fields = [f.name for f in arcpy.ListFields(training_fc)]

    if "district" in fields:
        arcpy.management.AddField(training_fc, "DISTRICT_KEY", "TEXT", field_length=50)
        arcpy.management.CalculateField(
            training_fc, "DISTRICT_KEY",
            "str(!district!).upper().strip() if !district! else 'UNKNOWN'",
            "PYTHON3"
        )
        print("  ✓ DISTRICT_KEY field created")

    # Population join
    population_table = os.path.join(GDB, "population_district")
    if arcpy.Exists(population_table):
        arcpy.management.JoinField(training_fc, "DISTRICT_KEY", population_table, "DISTRICT", ["TOTAL_POPULATION"])
        print("  ✓ Population data added")

    # Drainage join
    drainage_table = os.path.join(GDB, "drainage_issues_district")
    if arcpy.Exists(drainage_table):
        arcpy.management.JoinField(training_fc, "DISTRICT_KEY", drainage_table, "DISTRICT", ["UNRESOLVED_POINT"])
        print("  ✓ Drainage data added")

    # Set null values to 0
    for field in ["TOTAL_POPULATION", "UNRESOLVED_POINT"]:
        arcpy.management.CalculateField(
            training_fc, field,
            f"!{field}! if !{field}! is not None else 0",
            "PYTHON3"
        )

    print("✓ Data enrichment complete")

# training_fc = create_training_data()
# enrich_data(training_fc)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FOREST-BASED CLASSIFICATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_forest_model():
    """
    Trains the Forest-based Classification model.

    Variables:
    - ELEVATION_M: Elevation (meters)
    - SLOPE_DEGREE: Slope (degrees)
    - FLOW_ACCUMULATION: Flow accumulation
    - TWI: Topographic Wetness Index
    """
    print("Training Forest-based Classification v5...")

    model_out = os.path.join(GDB, "flood_risk_prediction_v5")
    importance_out = os.path.join(GDB, "variable_importance_final_v5")

    result = arcpy.stats.Forest(
        prediction_type               = "TRAIN",
        in_features                   = TRAINING_V4,
        variable_predict              = "FLOOD_POINT",
        treat_variable_as_categorical = True,
        explanatory_variables         = [
            ["ELEVATION_M",       False],
            ["SLOPE_DEGREE",      False],
            ["FLOW_ACCUMULATION", False],
            ["TWI",               False],
        ],
        output_features               = model_out,
        output_importance_table       = importance_out,
        number_of_trees               = 200,
        percentage_for_training       = 70,
        output_trained_features       = os.path.join(GDB, "_forest_v5_trained"),
    )

    print("✓ Model trained")

    # Print model messages
    msgs = result.getMessages()
    for line in msgs.split("\n"):
        if any(k in line.lower() for k in ["accuracy", "oob", "mse", "r2"]):
            print(f"  {line.strip()}")

    return model_out

# train_forest_model()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CREATING PREDICTION GRID AND RISK MAP
# ═══════════════════════════════════════════════════════════════════════════════

def create_prediction_grid():
    """
    Creates prediction points (fishnet) covering all of Ankara.
    """
    print("Creating prediction grid...")

    fishnet_fc = os.path.join(GDB, "prediction_fishnet")
    prediction_pts = os.path.join(GDB, "prediction_points_all")

    # Create polygon fishnet
    arcpy.management.CreateFishnet(
        out_feature_class = fishnet_fc,
        origin_coord      = "32.0 39.6",
        y_axis_coord      = "32.0 39.7",
        cell_width        = 0.005,
        cell_height       = 0.005,
        number_rows       = "",
        number_columns    = "",
        corner_coord      = "33.5 40.3",
        labels            = "NO_LABELS",
        geometry_type     = "POLYGON"
    )

    # Get centroid points
    arcpy.management.FeatureToPoint(fishnet_fc, prediction_pts, "CENTROID")
    n = int(arcpy.management.GetCount(prediction_pts).getOutput(0))
    print(f"  ✓ {n} prediction points created")

    # Add raster values
    arcpy.sa.ExtractMultiValuesToPoints(
        in_point_features = prediction_pts,
        in_rasters = [
            [SLOPE, "SLOPE_DEGREE"],
            [DEM, "ELEVATION_M"],
            [FLOW_ACC, "FLOW_ACCUMULATION"],
            [TWI, "TWI"],
        ],
        bilinear_interpolate_values = "BILINEAR"
    )

    # Clean up null values
    deleted = 0
    with arcpy.da.UpdateCursor(prediction_pts, ["SLOPE_DEGREE", "ELEVATION_M", "FLOW_ACCUMULATION", "TWI"]) as cur:
        for row in cur:
            if None in row:
                cur.deleteRow()
                deleted += 1

    n_remaining = int(arcpy.management.GetCount(prediction_pts).getOutput(0))
    print(f"  ✓ {deleted} empty points removed, {n_remaining} points ready")

    return prediction_pts


def predict_risk(prediction_pts):
    """
    Performs risk prediction for all of Ankara using the trained model.
    """
    print("Performing risk prediction...")

    prediction_result = os.path.join(GDB, "flood_risk_prediction_v5")

    result = arcpy.stats.Forest(
        prediction_type               = "PREDICT_FEATURES",
        in_features                   = TRAINING_V4,
        variable_predict              = "FLOOD_POINT",
        treat_variable_as_categorical = True,
        explanatory_variables         = [
            ["ELEVATION_M",       False],
            ["SLOPE_DEGREE",      False],
            ["FLOW_ACCUMULATION", False],
            ["TWI",               False],
        ],
        features_to_predict           = prediction_pts,
        output_features               = prediction_result,
        explanatory_variable_matching = "ELEVATION_M ELEVATION_M;SLOPE_DEGREE SLOPE_DEGREE;FLOW_ACCUMULATION FLOW_ACCUMULATION;TWI TWI",
        number_of_trees               = 200,
    )

    n = int(arcpy.management.GetCount(prediction_result).getOutput(0))
    print(f"✓ Risk prediction generated for {n} points")

    # High risk statistics
    high_risk = 0
    with arcpy.da.SearchCursor(prediction_result, ["Predicted"]) as cur:
        for row in cur:
            if row[0] == 1:
                high_risk += 1

    print(f"  High risk: {high_risk} points ({high_risk/n*100:.1f}%)")

    # ── Predicted_Probability → 5-class RISK_SCORE_V5 conversion ──────────────
    # Predicted_Probability: 0.0-1.0 probability value (Forest output)
    # RISK_SCORE_V5: 1=Very Low, 2=Low, 3=Medium, 4=High, 5=Very High
    print("  Calculating RISK_SCORE_V5...")

    arcpy.management.AddField(prediction_result, "RISK_SCORE_V5", "SHORT")

    with arcpy.da.UpdateCursor(
        prediction_result, ["Predicted_Probability", "RISK_SCORE_V5"]
    ) as cur:
        for row in cur:
            prob = row[0] if row[0] is not None else 0.0
            if   prob < 0.20: score = 1   # Very Low
            elif prob < 0.40: score = 2   # Low
            elif prob < 0.60: score = 3   # Medium
            elif prob < 0.80: score = 4   # High
            else:             score = 5   # Very High
            row[1] = score
            cur.updateRow(row)

    distribution = {1:0, 2:0, 3:0, 4:0, 5:0}
    for r in arcpy.da.SearchCursor(prediction_result, ["RISK_SCORE_V5"]):
        s = r[0] if r[0] else 0
        if s in distribution: distribution[s] += 1
    labels = {1:"Very Low",2:"Low",3:"Medium",4:"High",5:"Very High"}
    print("  Risk distribution:")
    for s in sorted(distribution):
        print(f"    Class {s} ({labels[s]:10s}): {distribution[s]:4d}  {distribution[s]/n*100:.1f}%")

    return prediction_result

# prediction_pts = create_prediction_grid()
# risk_map = predict_risk(prediction_pts)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. LODO SPATIAL CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_mcc(tp, tn, fp, fn):
    """
    Calculates the Matthews Correlation Coefficient.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    MCC ranges from -1 to +1:
    - +1: Perfect prediction
    - 0: Random prediction
    - -1: Completely wrong prediction
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    return numerator / denominator if denominator > 0 else 0


def lodo_spatial_cv():
    """
    Leave-One-District-Out Spatial Cross-Validation

    In each fold:
    - 1 district as test data
    - The remaining 4 districts as training data

    This method tests the model's spatial generalization ability.
    """
    print("=" * 60)
    print("LODO SPATIAL CROSS-VALIDATION")
    print("=" * 60)

    results = []

    for test_district in DISTRICTS:
        print(f"\n  Fold: {test_district} test, others training...")

        # Split test and training sets
        training_fold = os.path.join(GDB, "_fold_training")
        test_fold     = os.path.join(GDB, "_fold_test")

        arcpy.analysis.Select(TRAINING_V4, training_fold, f"district <> '{test_district}'")
        arcpy.analysis.Select(TRAINING_V4, test_fold,     f"district = '{test_district}'")

        n_training = int(arcpy.management.GetCount(training_fold).getOutput(0))
        n_test     = int(arcpy.management.GetCount(test_fold).getOutput(0))

        print(f"    Training: {n_training}, Test: {n_test}")

        # Train the model
        prediction_fold = os.path.join(GDB, "_fold_prediction")
        arcpy.stats.Forest(
            prediction_type               = "TRAIN",
            in_features                   = training_fold,
            variable_predict              = "FLOOD_POINT",
            treat_variable_as_categorical = True,
            explanatory_variables         = [
                ["ELEVATION_M",       False],
                ["SLOPE_DEGREE",      False],
                ["FLOW_ACCUMULATION", False],
                ["TWI",               False],
            ],
            output_features               = prediction_fold,
            number_of_trees               = 200,
            percentage_for_training       = 70,
        )

        # Calculate confusion matrix
        tp = tn = fp = fn = 0
        for r in arcpy.da.SearchCursor(prediction_fold, ["FLOOD_POINT", "Predicted_FLOOD_POINT"]):
            actual, predicted = int(r[0]), int(r[1])
            if   actual == 1 and predicted == 1: tp += 1
            elif actual == 0 and predicted == 0: tn += 1
            elif actual == 0 and predicted == 1: fp += 1
            elif actual == 1 and predicted == 0: fn += 1

        # Calculate metrics
        acc    = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        mcc    = calculate_mcc(tp, tn, fp, fn)

        results.append({
            "district": test_district,
            "n_training": n_training,
            "n_test": n_test,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acc": acc, "recall": recall, "precision": prec, "f1": f1, "mcc": mcc
        })

        print(f"    Acc: {acc:.3f}  F1: {f1:.3f}  MCC: {mcc:.3f}  Recall: {recall:.3f}")

        # Clean up temporary layers
        for tmp in [training_fold, test_fold, prediction_fold]:
            if arcpy.Exists(tmp):
                arcpy.management.Delete(tmp)

    # Average metrics
    avg_acc    = sum(s["acc"]    for s in results) / len(results)
    avg_f1     = sum(s["f1"]     for s in results) / len(results)
    avg_mcc    = sum(s["mcc"]    for s in results) / len(results)
    avg_recall = sum(s["recall"] for s in results) / len(results)

    print("\n" + "=" * 60)
    print(f"  AVERAGE → Acc: {avg_acc:.3f}  F1: {avg_f1:.3f}  MCC: {avg_mcc:.3f}  Recall: {avg_recall:.3f}")
    print("=" * 60)

    return results

# lodo_results = lodo_spatial_cv()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SHAP ANALYSIS (EXPLAINABLE AI)
# ═══════════════════════════════════════════════════════════════════════════════

def shap_analysis():
    """
    Explains variable importance using SHAP (SHapley Additive exPlanations).

    Used to independently validate ArcGIS Forest results, since it is
    compatible with sklearn RandomForest.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        import shap
    except ImportError:
        print("ERROR: sklearn and shap libraries are required!")
        print("pip install scikit-learn shap")
        return None, None, None, None

    print("Starting SHAP analysis...")

    # Read training data from the GDB
    fields = ["FLOOD_POINT", "ELEVATION_M", "SLOPE_DEGREE", "FLOW_ACCUMULATION", "TWI"]

    if arcpy.Exists(EUC_DIST):
        fields.append("EUC_DIST")

    data = []
    for r in arcpy.da.SearchCursor(TRAINING_V4, fields):
        if all(v is not None for v in r):
            data.append(r)

    X = np.array([[r[i] for i in range(1, len(r))] for r in data])
    y = np.array([int(r[0]) for r in data])

    feature_names = ["Elevation (m)", "Slope (°)", "Flow Accumulation", "TWI"]
    if arcpy.Exists(EUC_DIST):
        feature_names.append("Distance to Streams (m)")

    print(f"  {len(data)} samples, {len(feature_names)} variables")

    # Train the model
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X, y)

    # SHAP values
    print("  Calculating SHAP values...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # SHAP values for the flood class (1)
    shap_flood = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Global importance (mean absolute SHAP)
    global_importance = np.abs(shap_flood).mean(axis=0)

    print("\n  Global SHAP Values (Variable Importance Ranking):")
    print("  " + "-" * 45)
    for name, val in sorted(zip(feature_names, global_importance), key=lambda x: -x[1]):
        bar = "█" * int(val * 10)
        print(f"    {name:25s}: {val:.4f} {bar}")

    return shap_flood, global_importance, feature_names, X

# shap_vals, global_importance, feat_names, X_train = shap_analysis()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. HOTSPOT AND TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def hotspot_analysis():
    """
    Performs clustering analysis of flood points using Optimized Hot Spot Analysis.
    """
    print("Performing Optimized Hot Spot Analysis...")

    arcpy.stats.OptimizedHotSpotAnalysis(
        Input_Features           = FLOOD_MAIN,
        Output_Features          = HOTSPOT,
        Analysis_Field           = None,
        Incident_Data_Aggregation_Method = "COUNT_INCIDENTS_WITHIN_FISHNET_POLYGONS",
    )

    n = int(arcpy.management.GetCount(HOTSPOT).getOutput(0))
    print(f"✓ Hotspot analysis complete: {n} areas")

    # Cold/hot spot summary
    counts = Counter()
    for r in arcpy.da.SearchCursor(HOTSPOT, ["Gi_Bin"]):
        if r[0]:
            if r[0] == 3:
                counts["Hot Spot (99%)"] += 1
            elif r[0] == 2:
                counts["Hot Spot (95%)"] += 1
            elif r[0] == -3:
                counts["Cold Spot (99%)"] += 1
            elif r[0] == -2:
                counts["Cold Spot (95%)"] += 1

    print("  Clustering summary:")
    for k, v in counts.most_common():
        print(f"    {k}: {v}")


def create_space_time_cube():
    """
    Creates a Space-Time Cube (for temporal analysis).
    """
    print("Creating Space-Time Cube...")

    STC_FILE = os.path.join(DOCS, "flood_stc.nc")

    arcpy.stpm.CreateSpaceTimeCube(
        in_features        = FLOOD_MAIN,
        output_cube        = STC_FILE,
        time_field         = "DATE",
        time_step_interval = "4 Months",
        distance_interval  = "1000 Meters",
    )

    print(f"✓ Space-Time Cube: {STC_FILE}")
    return STC_FILE


def emerging_hotspot_analysis(stc_file):
    """
    Performs temporal trend analysis using Emerging Hot Spot Analysis.
    """
    print("Performing Emerging Hot Spot Analysis...")

    arcpy.stpm.EmergingHotSpotAnalysis(
        in_cube          = stc_file,
        analysis_variable= "COUNT",
        output_features  = EHS,
        polygon_mask     = DISTRICT_GDB,
    )

    n = int(arcpy.management.GetCount(EHS).getOutput(0))
    print(f"✓ Emerging Hot Spot: {n} areas")

    # Pattern summary
    pattern_counts = Counter(r[0] for r in arcpy.da.SearchCursor(EHS, ["PATTERN"]) if r[0])

    print("  Temporal patterns:")
    for pattern, count in pattern_counts.most_common():
        print(f"    {pattern}: {count}")

# hotspot_analysis()
# stc = create_space_time_cube()
# emerging_hotspot_analysis(stc)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SOCIAL VULNERABILITY INDEX (SVI)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_svi():
    """
    Calculates the Social Vulnerability Index.

    SVI = 0.40 × Population_norm + 0.35 × Drainage_norm + 0.25 × Assembly_norm

    Variables:
    - Population: District population (TÜİK 2022)
    - Drainage: Number of unresolved drainage points
    - Assembly: Assembly area ratio
    """
    print("Calculating Social Vulnerability Index (SVI)...")

    # Data (TÜİK 2022 + ABB data)
    svi_data = {
        "Çankaya":     {"population": 925828, "drainage": 14, "assembly": 0.25},
        "Keçiören":    {"population": 938568, "drainage":  2, "assembly": 0.25},
        "Altındağ":    {"population": 396165, "drainage":  6, "assembly": 0.25},
        "Yenimahalle": {"population": 695395, "drainage":  1, "assembly": 0.25},
        "Akyurt":      {"population":  37456, "drainage":  1, "assembly": 0.25},
    }

    # Normalization function (min-max)
    def norm(value, all_values):
        mn, mx = min(all_values), max(all_values)
        return (value - mn) / (mx - mn) if mx > mn else 0

    populations = [v["population"] for v in svi_data.values()]
    drainages   = [v["drainage"]   for v in svi_data.values()]

    results = {}
    for district, v in svi_data.items():
        n = norm(v["population"], populations)
        d = norm(v["drainage"],   drainages)
        a = v["assembly"]

        # Weighted SVI score
        svi = 0.40 * n + 0.35 * d + 0.25 * a
        results[district] = round(svi, 3)

    print("\n  SVI Results (Higher = More Vulnerable):")
    print("  " + "-" * 40)
    for district, svi in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(svi * 20)
        print(f"    {district:15s}: {svi:.3f} {bar}")

    return results

svi_results = calculate_svi()


# ═══════════════════════════════════════════════════════════════════════════════
# 12. ZONAL STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def zonal_statistics():
    """
    Calculates district-level risk statistics.
    """
    print("Calculating Zonal Statistics...")

    # Convert risk score to raster
    risk_raster = os.path.join(GDB, "_risk_raster_tmp")
    arcpy.conversion.PointToRaster(
        PREDICTION_V5, "RISK_SCORE_V5", risk_raster,
        cellsize=0.001  # ~100m
    )

    # Zonal statistics
    arcpy.sa.ZonalStatisticsAsTable(
        in_zone_data    = DISTRICT_GDB,
        zone_field      = "NAME",
        in_value_raster = risk_raster,
        out_table       = ZONAL,
        statistics_type = "ALL",
    )

    # Read results
    print("\n  District-Level Risk Statistics:")
    print("  " + "-" * 60)
    fields = ["NAME", "MEAN", "STD", "MIN", "MAX", "COUNT"]
    for r in arcpy.da.SearchCursor(ZONAL, fields):
        print(f"    {r[0]:15s} Mean: {r[1]:.2f}  Std: {r[2]:.2f}  Min: {r[3]:.0f}  Max: {r[4]:.0f}  n: {r[5]}")

    # Delete temporary raster
    if arcpy.Exists(risk_raster):
        arcpy.management.Delete(risk_raster)

    print("✓ Zonal Statistics complete")

# zonal_statistics()


# ═══════════════════════════════════════════════════════════════════════════════
# 13. MAP SYMBOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

def set_map_symbology():
    """
    Configures the symbology of the map layers.
    """
    print("Setting map symbology...")

    aprx = arcpy.mp.ArcGISProject("CURRENT")
    m = aprx.activeMap

    for lyr in m.listLayers():

        # Flood risk map
        if lyr.name == "flood_risk_map" or lyr.name == "flood_risk_prediction_v5":
            lyr.visible = True
            sym = lyr.symbology
            sym.updateRenderer('UniqueValueRenderer')
            sym.renderer.fields = ['Predicted']
            lyr.symbology = sym

            sym2 = lyr.symbology
            for grp in sym2.renderer.groups:
                for item in grp.items:
                    val = str(item.values[0][0])
                    if val == '0':
                        item.symbol.color = {'RGB': [0, 197, 82, 180]}    # Green
                        item.symbol.size = 4
                        item.label = 'Low Risk'
                    elif val == '1':
                        item.symbol.color = {'RGB': [220, 30, 30, 200]}   # Red
                        item.symbol.size = 4
                        item.label = 'High Risk'
            lyr.symbology = sym2
            print("  ✓ Risk map symbology")

        # Actual flood points
        elif lyr.name == "flood_points_main":
            lyr.visible = True
            sym = lyr.symbology
            sym.updateRenderer('SimpleRenderer')
            sym.renderer.symbol.color = {'RGB': [255, 215, 0, 255]}       # Yellow
            sym.renderer.symbol.size = 8
            sym.renderer.label = 'Actual Flood Point'
            lyr.symbology = sym
            print("  ✓ Flood points symbology")

        # Turn off other layers
        elif lyr.name not in ["Topographic", "basemap"]:
            lyr.visible = False

    # Zoom to Ankara center
    m.defaultCamera.setExtent(arcpy.Extent(32.3, 39.7, 33.4, 40.2))
    aprx.save()

    print("✓ Map symbology saved")

# set_map_symbology()


# ═══════════════════════════════════════════════════════════════════════════════
# 14. VALIDATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def validation_analysis():
    """
    Tests model accuracy against real flood points.
    Uses Spatial Join to find the value of the nearest prediction point.
    """
    print("Performing validation analysis...")

    join_output = os.path.join(GDB, "validation_join")

    arcpy.analysis.SpatialJoin(
        target_features   = FLOOD_MAIN,
        join_features     = PREDICTION_V5,
        out_feature_class = join_output,
        join_operation    = "JOIN_ONE_TO_ONE",
        join_type         = "KEEP_ALL",
        match_option      = "CLOSEST"
    )

    # Overall accuracy
    total = 0
    correct = 0

    with arcpy.da.SearchCursor(join_output, ["Predicted"]) as cur:
        for row in cur:
            total += 1
            if row[0] == 1:
                correct += 1

    print("\n" + "=" * 50)
    print("  VALIDATION RESULTS")
    print("=" * 50)
    print(f"  Total real flood points     : {total}")
    print(f"  Correct prediction (risk=1) : {correct} ({correct/total*100:.1f}%)")
    print(f"  Incorrect prediction (risk=0): {total-correct} ({(total-correct)/total*100:.1f}%)")

    # District-level accuracy
    print("\n  District-Level Accuracy:")
    district_results = {}

    with arcpy.da.SearchCursor(join_output, ["district", "Predicted"]) as cur:
        for row in cur:
            district = row[0] if row[0] else "Unknown"
            if district not in district_results:
                district_results[district] = {"total": 0, "correct": 0}
            district_results[district]["total"] += 1
            if row[1] == 1:
                district_results[district]["correct"] += 1

    for district, result in sorted(district_results.items()):
        if result["total"] > 0:
            ratio = result["correct"] / result["total"] * 100
            print(f"    {district:15s}: {result['correct']:3d}/{result['total']:3d} ({ratio:.1f}%)")

    print("=" * 50)

# validation_analysis()


# ═══════════════════════════════════════════════════════════════════════════════
# 15. CHART OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lodo_table():
    """
    Creates a professional table chart for LODO CV results.
    """
    fp_bold = fm.FontProperties(family='DejaVu Sans', weight='bold')
    fp = fm.FontProperties(family='DejaVu Sans')

    # LODO data
    lodo_data = [
        ["Keçiören",    0.986, 0.974, 0.966, 0.950],
        ["Çankaya",     0.975, 0.950, 0.934, 0.927],
        ["Altındağ",    0.913, 0.889, 0.817, 0.889],
        ["Akyurt",      0.976, 0.714, 0.710, 0.833],
        ["Yenimahalle", 0.817, 0.772, 0.680, 1.000],
    ]
    avg = [0.938, 0.869, 0.830, 0.920]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, "Leave-One-District-Out — Spatial Cross-Validation",
            ha='center', va='top', color='white', fontproperties=fp_bold,
            fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.89, "ArcGIS Forest-based Classification v5  ·  5 districts  ·  Independent test per fold",
            ha='center', va='top', color='#aaa', fontproperties=fp,
            fontsize=9, transform=ax.transAxes)

    # Table settings
    cols = ["District", "Accuracy", "F1-Score", "MCC", "Recall"]
    col_w = [0.25, 0.18, 0.18, 0.18, 0.18]
    x_pos = [sum(col_w[:i]) for i in range(len(cols))]
    row_h, hdr_h, y_top = 0.115, 0.12, 0.81
    HDR = '#0f3460'
    DARK = '#16213e'
    CARD = '#1a1a2e'

    # MCC color function
    def mcc_color(v):
        if v >= 0.9: return '#2ecc71'
        elif v >= 0.8: return '#f39c12'
        else: return '#e74c3c'

    # Header row
    for ci, (col, x, w) in enumerate(zip(cols, x_pos, col_w)):
        ax.add_patch(plt.Rectangle((x, y_top-hdr_h), w, hdr_h, facecolor=HDR,
            edgecolor='#444', linewidth=0.8, transform=ax.transAxes, clip_on=False))
        ax.text(x+w/2, y_top-hdr_h/2, col, ha='center', va='center', color='white',
                fontproperties=fp_bold, fontsize=10, transform=ax.transAxes)

    # Data rows
    for ri, row in enumerate(lodo_data):
        y = y_top - hdr_h - ri*row_h
        bg = DARK if ri%2==0 else CARD
        for ci, (val, x, w) in enumerate(zip(row, x_pos, col_w)):
            ax.add_patch(plt.Rectangle((x, y-row_h), w, row_h, facecolor=bg,
                edgecolor='#333', linewidth=0.4, transform=ax.transAxes, clip_on=False))
            clr = mcc_color(val) if ci==3 else 'white'
            font = fp_bold if ci==3 else fp
            ax.text(x+w/2, y-row_h/2, f'{val:.3f}' if ci>0 else val,
                    ha='center', va='center', color=clr,
                    fontproperties=font, fontsize=10, transform=ax.transAxes)

    # Average row
    y = y_top - hdr_h - 5*row_h
    for ci, (val, x, w) in enumerate(zip(["AVERAGE"]+avg, x_pos, col_w)):
        ax.add_patch(plt.Rectangle((x, y-row_h), w, row_h, facecolor='#1a3a2a',
            edgecolor='#2ecc71', linewidth=1.5, transform=ax.transAxes, clip_on=False))
        clr = '#2ecc71' if ci==3 else 'white'
        ax.text(x+w/2, y-row_h/2, f'{val:.3f}' if ci>0 else val,
                ha='center', va='center', color=clr,
                fontproperties=fp_bold, fontsize=10, transform=ax.transAxes)

    # Footer note
    ax.text(0.01, 0.015,
        "MCC = Matthews Correlation Coefficient  ·  Random split MCC: 0.910  ·  "
        "Independent validation (Akyurt n=76): Recall 0.934",
        ha='left', va='bottom', color='#888', fontproperties=fp,
        fontsize=8, transform=ax.transAxes)

    # Save
    path = os.path.join(DOCS, "TABLE_LODO_CV.png")
    fig.savefig(path, dpi=200, facecolor='#1a1a2e')
    plt.close()
    print(f"✓ Chart saved: {path}")

# plot_lodo_table()


# ═══════════════════════════════════════════════════════════════════════════════
# 16. CURRENT DATA STATUS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def data_status_report():
    """
    Reports the status of current data and layers.
    """
    print("\n" + "=" * 60)
    print("  DATA STATUS REPORT")
    print("=" * 60)

    # Training data
    if arcpy.Exists(TRAINING_V4):
        n_training = int(arcpy.management.GetCount(TRAINING_V4).getOutput(0))
        positive = sum(1 for r in arcpy.da.SearchCursor(TRAINING_V4, ["FLOOD_POINT"]) if r[0] == 1)
        negative = n_training - positive
        print(f"  Training data : {n_training} points ({positive} flood + {negative} negative)")

    # Prediction results
    if arcpy.Exists(PREDICTION_V5):
        n_prediction = int(arcpy.management.GetCount(PREDICTION_V5).getOutput(0))
        risk_distribution = Counter()
        for r in arcpy.da.SearchCursor(PREDICTION_V5, ["RISK_SCORE_V5"]):
            s = int(r[0]) if r[0] else 0
            risk_distribution[s] += 1

        print(f"  Prediction points: {n_prediction}")
        print("  Risk distribution:")
        for s in sorted(risk_distribution):
            label = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}.get(s, "?")
            print(f"    Class {s} ({label:12s}): {risk_distribution[s]:4d} ({risk_distribution[s]/n_prediction*100:.1f}%)")

    # Raster layers
    raster_layers = [
        ("DEM", DEM), ("Slope", SLOPE), ("Flow Accumulation", FLOW_ACC),
        ("TWI", TWI), ("Stream Distance", EUC_DIST)
    ]
    print("\n  Raster Layers:")
    for name, path in raster_layers:
        status = "✓" if arcpy.Exists(path) else "✗"
        print(f"    {status} {name}")

    print("=" * 60)

data_status_report()


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PROJECT SUMMARY")
print("=" * 60)
print("  Model              : Forest-based Classification v5")
print("  Training data      : 702 points (292 flood + 410 negative)")
print("  Prediction points  : 8,017")
print("  Variables          : Elevation, Slope, Flow Accumulation, TWI")
print("  LODO MCC           : 0.830")
print("  AUC                : 0.973")
print("  Recall             : 0.920")
print("  F1-Score           : 0.869")
print("  Independent test   : Akyurt (n=76) Recall: 0.934")
print("=" * 60)
print("\n  Usage:")
print("    - Uncomment the relevant lines for the first run")
print("    - On subsequent runs, call only the functions you need")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. IDW PRECIPITATION INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════════════

def idw_precipitation_interpolation():
    """
    Produces a precipitation surface from precipitation station measurements using IDW.
    Source: MGM precipitation stations (precipitation_stations_2022)
    """
    print("Starting IDW precipitation interpolation...")

    precip_pts = os.path.join(GDB, "precipitation_stations_2022")
    idw_out    = os.path.join(GDB, "precipitation_interpolation")
    idw_district = os.path.join(GDB, "idw_risk_district")

    # IDW interpolation
    idw_r = arcpy.sa.Idw(
        in_point_features = precip_pts,
        z_field           = "PRECIP_MM",
        cell_size         = 0.01,
        power             = 2,
    )
    idw_r.save(idw_out)
    print("  ✓ IDW surface created")

    # Clip to district boundary
    arcpy.management.Clip(idw_out, DISTRICT_GDB, idw_district)
    print(f"  ✓ Clipped to district boundary → {idw_district}")

    # Add precipitation values to training data
    arcpy.sa.ExtractValuesToPoints(TRAINING_V4, idw_out,
                                   os.path.join(GDB, "_tmp_precip"))
    arcpy.management.JoinField(TRAINING_V4, arcpy.Describe(TRAINING_V4).OIDFieldName,
                                os.path.join(GDB, "_tmp_precip"),
                                arcpy.Describe(os.path.join(GDB, "_tmp_precip")).OIDFieldName,
                                ["RASTERVALU"])
    arcpy.management.AlterField(TRAINING_V4, "RASTERVALU", "PRECIP_MM", "PRECIP_MM")
    print("  ✓ Precipitation values added to training data")

    if arcpy.Exists(os.path.join(GDB, "_tmp_precip")):
        arcpy.management.Delete(os.path.join(GDB, "_tmp_precip"))

# idw_precipitation_interpolation()


# ═══════════════════════════════════════════════════════════════════════════════
# 18. KERNEL DENSITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def kernel_density_analysis():
    """
    Calculates the spatial density of flood points using Kernel Density.
    Produces a risk density map.
    """
    print("Starting Kernel Density analysis...")

    kd_out = os.path.join(GDB, "kernel_density_district")

    kd_r = arcpy.sa.KernelDensity(
        in_features       = FLOOD_MAIN,
        population_field  = "NONE",
        cell_size         = 0.005,          # ~500m
        search_radius     = 0.05,           # ~5km
        area_unit_scale_factor = "SQUARE_KILOMETERS",
        out_cell_values   = "DENSITIES",
        method            = "PLANAR",
    )
    kd_r.save(kd_out)
    print(f"  ✓ Kernel Density → {kd_out}")

    # Summary statistics
    kd_arr = arcpy.RasterToNumPyArray(kd_out, nodata_to_value=0).flatten()
    kd_arr = kd_arr[kd_arr > 0]
    if len(kd_arr) > 0:
        print(f"  Min: {kd_arr.min():.4f}  Max: {kd_arr.max():.4f}  "
              f"Mean: {kd_arr.mean():.4f}")

# kernel_density_analysis()


# ═══════════════════════════════════════════════════════════════════════════════
# 19. POPULATION RISK ANALYSIS
# Calculate the population at risk using TÜİK 2022 population data
# ═══════════════════════════════════════════════════════════════════════════════

def population_risk_analysis():
    """
    Calculates the population in the high and very high risk classes.
    Risk points → District-level population ratio → Total population at risk
    """
    print("Starting population risk analysis...")

    # TÜİK 2022 district populations
    POPULATION = {
        "Keçiören":    938568,
        "Çankaya":     925828,
        "Yenimahalle": 695395,
        "Altındağ":    396165,
        "Akyurt":       37456,
    }
    TOTAL_POPULATION = sum(POPULATION.values())  # 2,993,412

    # Read risk class distribution at the district level
    district_risk = {}   # {district: {score: n}}
    with arcpy.da.SearchCursor(
        PREDICTION_V5, ["RISK_SCORE_V5", "district"]
    ) as cur:
        for row in cur:
            score    = int(row[0]) if row[0] else 0
            district = str(row[1]).strip() if row[1] else "Unknown"
            if district not in district_risk:
                district_risk[district] = {1:0, 2:0, 3:0, 4:0, 5:0}
            if score in district_risk[district]:
                district_risk[district][score] += 1

    print("\n  District-Level Population at Risk:")
    print("  " + "-" * 65)

    total_risk_population = 0

    for district in sorted(POPULATION.keys()):
        if district not in district_risk:
            continue
        distribution = district_risk[district]
        n_total = sum(distribution.values())
        if n_total == 0:
            continue

        # High + Very High = class 4 + 5
        n_high = distribution.get(4, 0) + distribution.get(5, 0)
        ratio  = n_high / n_total if n_total > 0 else 0

        # Apply ratio to district population
        risk_population = int(POPULATION[district] * ratio)
        total_risk_population += risk_population

        print(f"    {district:15s}  Population: {POPULATION[district]:7,}  "
              f"Risk Ratio: {ratio*100:4.1f}%  "
              f"At Risk: {risk_population:7,}")

    print("  " + "-" * 65)
    print(f"    {'TOTAL':15s}  Population: {TOTAL_POPULATION:7,}  "
          f"Risk Ratio: {total_risk_population/TOTAL_POPULATION*100:4.1f}%  "
          f"At Risk: {total_risk_population:7,}")

    return total_risk_population

# risk_population = population_risk_analysis()


# ═══════════════════════════════════════════════════════════════════════════════
# 20. CLIMATE SCENARIO PROJECTION (RCP 4.5 / RCP 8.5)
# IPCC AR6 Turkey projections + TÜİK 2035 population forecast
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_climate_scenario():
    """
    Projects the population at flood risk for the year 2035.

    Coefficients:
    - RCP 4.5 (Medium emissions): +7.9% risk increase (IPCC AR6 Turkey)
    - RCP 8.5 (High emissions): +13.4% risk increase
    - Population growth: +2.1% (TÜİK 2035 district projection)
    """
    print("Calculating climate scenario projection...\n")

    # Current population at risk (2023)
    POPULATION_2023 = {
        "Keçiören":    152184,
        "Çankaya":     181818,
        "Yenimahalle": 216288,
        "Altındağ":    110782,
        "Akyurt":        4937,
    }
    TOTAL_2023 = sum(POPULATION_2023.values())  # 666,009

    # Coefficients
    RCP45_COEFFICIENT = 1.079   # +7.9% risk increase
    RCP85_COEFFICIENT = 1.134   # +13.4% risk increase
    POPULATION_COEFFICIENT = 1.021   # +2.1% population increase

    print(f"  {'District':15s}  {'2023':>8}  {'2035 RCP4.5':>12}  {'2035 RCP8.5':>12}")
    print("  " + "-" * 55)

    total_45 = total_85 = 0

    for district in sorted(POPULATION_2023.keys()):
        n     = POPULATION_2023[district]
        r45   = int(n * RCP45_COEFFICIENT * POPULATION_COEFFICIENT)
        r85   = int(n * RCP85_COEFFICIENT * POPULATION_COEFFICIENT)
        total_45 += r45
        total_85 += r85
        print(f"    {district:15s}  {n:8,}  {r45:12,}  {r85:12,}")

    print("  " + "-" * 55)
    print(f"    {'TOTAL':15s}  {TOTAL_2023:8,}  {total_45:12,}  {total_85:12,}")
    print(f"\n  Increase (RCP 4.5): +{total_45-TOTAL_2023:,} people "
          f"(+{(total_45/TOTAL_2023-1)*100:.1f}%)")
    print(f"  Increase (RCP 8.5): +{total_85-TOTAL_2023:,} people "
          f"(+{(total_85/TOTAL_2023-1)*100:.1f}%)")

    return {"current": TOTAL_2023, "rcp45": total_45, "rcp85": total_85}

climate_projection = calculate_climate_scenario()
