import ee

# ------------------------------------------------------------
# 0. Initialize Earth Engine
# ------------------------------------------------------------
try:
    ee.Initialize(project="geo-uhi")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="geo-uhi")

# ------------------------------------------------------------
# 1. AOI: Central Bangalore (~270 km²)
# ------------------------------------------------------------
AOI = ee.Geometry.Rectangle(
    [77.55, 12.90, 77.70, 13.05], geodesic=False
)

# Years for temporal comparison
YEARS = range(2020, 2026)

# Seasonal windows (India-appropriate)
SEASONS = [
    ("summer", "-03-01", "-05-15"),      # pre-monsoon / peak UHI
    ("winter", "-11-15", "-01-15")       # cool season
]

# ------------------------------------------------------------
# 2. LANDSAT-9 LST (Primary UHI Signal)
# ------------------------------------------------------------
landsat9 = (
    ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(AOI)
)

def preprocess_landsat(img):
    qa = img.select("QA_PIXEL")

    # Cloud + shadow mask
    mask = (
        qa.bitwiseAnd(1 << 3).eq(0)   # cloud
        .And(qa.bitwiseAnd(1 << 4).eq(0))  # shadow
    )

    # USGS Collection-2 LST scaling (Kelvin)
    lst = (
        img.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .updateMask(img.select("ST_B10").gt(0))
        .rename("LST_K")
    )

    return (
        lst.updateMask(mask)
           .clip(AOI)
           .copyProperties(img, ["system:time_start"])
    )

# ------------------------------------------------------------
# 3. MODIS LST (Independent Validation)
# ------------------------------------------------------------
modis = (
    ee.ImageCollection("MODIS/061/MOD11A1")
    .filterBounds(AOI)
)

def preprocess_modis(img):
    return (
        img.select("LST_Day_1km")
           .multiply(0.02)        # MODIS scale factor
           .rename("LST_MODIS_K")
           .clip(AOI)
           .copyProperties(img, ["system:time_start"])
    )

# ------------------------------------------------------------
# 4. Seasonal Loop: 2020–2025
# ------------------------------------------------------------
for year in YEARS:
    for season, start_suffix, end_suffix in SEASONS:

        start_date = f"{year}{start_suffix}"
        end_date = f"{year+1}{end_suffix}" if season == "winter" else f"{year}{end_suffix}"

        # ---------- LANDSAT ----------
        l9_season = (
            landsat9
            .filterDate(start_date, end_date)
            .map(preprocess_landsat)
        )

        if l9_season.size().getInfo() > 0:
            l9_lst = l9_season.median()

            ee.batch.Export.image.toDrive(
                image=l9_lst,
                description=f"Bangalore_L9_LST_{season}_{year}",
                folder="Research_Data",
                region=AOI,
                scale=30,
                crs="EPSG:32643",   # metric CRS for UHI analysis
                maxPixels=1e10
            ).start()

            print(f"✅ Landsat-9 export started: {season} {year}")
        else:
            print(f"⚠️ No Landsat-9 data: {season} {year}")

        # ---------- MODIS ----------
        modis_season = (
            modis
            .filterDate(start_date, end_date)
            .map(preprocess_modis)
        )

        if modis_season.size().getInfo() > 0:
            modis_lst = modis_season.mean()

            ee.batch.Export.image.toDrive(
                image=modis_lst,
                description=f"Bangalore_MODIS_LST_{season}_{year}_Validation",
                folder="Research_Data",
                region=AOI,
                scale=1000,
                crs="EPSG:32643",
                maxPixels=1e10
            ).start()

            print(f"✅ MODIS export started: {season} {year}")
        else:
            print(f"⚠️ No MODIS data: {season} {year}")

print("\n🚀 All 5-year seasonal Landsat + MODIS exports submitted")
