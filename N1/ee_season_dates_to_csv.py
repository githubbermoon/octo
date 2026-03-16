import csv
import ee

# Initialize Earth Engine
try:
    ee.Initialize(project="geo-uhi")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="geo-uhi")

AOI = ee.Geometry.Rectangle([77.55, 12.90, 77.70, 13.05], geodesic=False)

YEARS = range(2020, 2026)
SEASONS = [
    ("summer", "-03-01", "-05-15"),
    ("winter", "-11-15", "-01-15"),
]

# ---- Sentinel-2 SR ----
def mask_s2(img):
    scl = img.select("SCL")
    mask = (
        scl.neq(3)
        .And(scl.neq(7))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )
    return img.updateMask(mask).copyProperties(img, ["system:time_start"])

s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .map(mask_s2)
)

# ---- Landsat-8 L2 ----
def preprocess_l8(img):
    qa = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(mask).copyProperties(img, ["system:time_start"])

l8 = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterBounds(AOI)
    .map(preprocess_l8)
)

# ---- Landsat-9 L2 ----
def preprocess_l9(img):
    qa = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(mask).copyProperties(img, ["system:time_start"])

l9 = (
    ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(AOI)
    .map(preprocess_l9)
)

# ---- MODIS LST ----
modis = (
    ee.ImageCollection("MODIS/061/MOD11A1")
    .filterBounds(AOI)
)

SENSORS = {
    "Sentinel-2": s2,
    "Landsat-8": l8,
    "Landsat-9": l9,
    "MODIS": modis,
}


def list_dates(coll, start, end):
    coll = coll.filterDate(start, end)
    ts = coll.aggregate_array("system:time_start")
    dates = ee.List(ts).map(lambda t: ee.Date(t).format("YYYY-MM-dd")).distinct()
    dates_list = dates.getInfo()
    dates_sorted = sorted(dates_list)
    median_date = dates_sorted[len(dates_sorted) // 2] if dates_sorted else ""
    return dates_sorted, median_date


def main():
    out_path = "season_dates.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "year",
            "season",
            "start_date",
            "end_date",
            "sensor",
            "scene_count",
            "median_date",
            "dates",
        ])

        for year in YEARS:
            for season, sfx_start, sfx_end in SEASONS:
                start = f"{year}{sfx_start}"
                end = f"{year + 1}{sfx_end}" if season == "winter" else f"{year}{sfx_end}"

                for name, coll in SENSORS.items():
                    dates, median_date = list_dates(coll, start, end)
                    writer.writerow([
                        year,
                        season,
                        start,
                        end,
                        name,
                        len(dates),
                        median_date,
                        ";".join(dates),
                    ])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
