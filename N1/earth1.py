import ee
ee.Authenticate()
ee.Initialize(project="geo-uhi")

# AOI
bangalore_box = ee.Geometry.Rectangle(
    [77.55, 12.90, 77.70, 13.05], geodesic=False
)

# Sentinel-2 SR
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(bangalore_box)
    # .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))  # 🔑 relaxed
)
print(f"Total images in AOI before date filtering: {s2.size().getInfo()}")

# Pixel-level cloud mask using SCL
def mask_s2(img):
    scl = img.select("SCL")
    mask = (
        scl.neq(3)   # cloud shadow
        .And(scl.neq(7))  # unclassified
        .And(scl.neq(8))  # cloud medium
        .And(scl.neq(9))  # cloud high
        .And(scl.neq(10)) # cirrus
    )

    img = ee.Image(
        img.updateMask(mask)
        .select(["B2", "B3", "B4", "B8"])
        .divide(10000)                     # normalization
        .clip(bangalore_box)
        .reproject("EPSG:4326", None, 10)  # alignment
        .copyProperties(img, ["system:time_start"]) # 🔑 Restore timestamp
    )

    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return img.addBands(ndvi)

s2 = s2.map(mask_s2)

years = range(2020, 2026)

tasks = []

for year in years:
    for season, start, end in [
        ("summer", f"{year}-03-01", f"{year}-05-15"),
        ("winter", f"{year}-11-15", f"{year+1}-01-15")
    ]:
        coll = s2.filterDate(start, end)

        # 🔑 SAFE check (server-side)
        count = coll.size().getInfo()
        if count == 0:
            print(f"⚠️ Skipping {season} {year}: no images after masking")
            continue

        img = coll.median().set({
            "season": season,
            "year": year
        })

        desc = f"Bangalore_S2_{season}_{year}"
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=desc,
            folder="Research_Data",
            region=bangalore_box,
            scale=10,
            crs="EPSG:4326",
            maxPixels=1e10
        )
        task.start()
        tasks.append(task)
        print(f"✅ Started export: {desc}")

# Monitor
print("\nWaiting for tasks to complete...")
for task in tasks:
    task.status()
