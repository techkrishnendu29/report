from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
import ee
import datetime
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# ---------------------- Initialize FastAPI ----------------------
app = FastAPI(title="Crop Health & Soil Report API")

# ---------------------- GEE Authentication ----------------------
serviceaccount = 'gee-backend@ee-bairagisayan464.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(serviceaccount,
                                           'ee-bairagisayan464-6b20e1402738.json')
ee.Initialize(credentials, project='ee-bairagisayan464')

# ---------------------- Function: Get Report Data ----------------------
def generate_report(state_name: str, district_name: str):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30)

    fc = ee.FeatureCollection("projects/ee-bairagisayan464/assets/GADM-IND")
    region = fc.filter(
        ee.Filter.And(
            ee.Filter.eq("NAME_1", state_name),
            ee.Filter.eq("NAME_2", district_name)
        )
    )

    def s2Mask(img):
        qa = img.select('QA60')
        mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(mask)

    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterDate(str(start), str(end))
                  .filterBounds(region)
                  .map(s2Mask))

    if collection.size().getInfo() == 0:
        return {"error": "No Sentinel-2 images found for this region/date range."}

    image = collection.median().clip(region)

    # Indices
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = image.expression(
        '2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
    ).rename('EVI')
    savi = image.expression(
        '((NIR-RED)/((NIR+RED+L)*(1+L)))',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'L': 0.5}
    ).rename('SAVI')
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    msi = image.expression('(SWIR/NIR)', {'SWIR': image.select('B11'), 'NIR': image.select('B8')}).rename('MSI')

    # Mean function
    def mean_index(img, name):
        val = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region.geometry(),
            scale=30,
            maxPixels=1e13,
            bestEffort=True
        ).get(name)
        return val.getInfo() if val else None

    indices = {
        'NDVI': mean_index(ndvi, 'NDVI'),
        'EVI': mean_index(evi, 'EVI'),
        'SAVI': mean_index(savi, 'SAVI'),
        'GNDVI': mean_index(gndvi, 'GNDVI'),
        'NDWI': mean_index(ndwi, 'NDWI'),
        'MSI': mean_index(msi, 'MSI')
    }

    # Soil condition summary
    soil_summary = ""
    if indices['SAVI'] and indices['SAVI'] < 0.3:
        soil_summary += "Very low vegetation cover (bare soil).\n"
    elif indices['SAVI'] and indices['SAVI'] < 0.6:
        soil_summary += "Moderate vegetation cover.\n"
    else:
        soil_summary += "Dense healthy vegetation.\n"

    if indices['MSI'] and indices['MSI'] > 1.5:
        soil_summary += "High moisture stress detected.\n"
    else:
        soil_summary += "Soil moisture is within healthy range.\n"

    if indices['NDWI'] and indices['NDWI'] < 0:
        soil_summary += "Low water presence, possible dry condition.\n"
    else:
        soil_summary += "Adequate water presence in soil.\n"

    # Crop health & pest risk
    crop_health = "Normal"
    pest_risk = "Low"
    pest_risk_percent = 0
    risk_score = 0

    if indices['NDVI'] and indices['NDVI'] < 0.3:
        risk_score += 30
    if indices['MSI'] and indices['MSI'] > 1.5:
        risk_score += 25
    if indices['NDWI'] and indices['NDWI'] < 0.2:
        risk_score += 15

    pest_risk_percent = min(100, risk_score)
    if pest_risk_percent < 30:
        pest_risk = "Low"
    elif pest_risk_percent < 60:
        pest_risk = "Moderate"
    else:
        pest_risk = "High"

    if indices['NDVI'] and indices['NDVI'] > 0.6 and indices['NDWI'] and indices['NDWI'] > 0:
        crop_health = "Healthy"
    elif indices['NDVI'] and indices['NDVI'] < 0.3:
        crop_health = "Poor / Stress detected"

    # Weather API
    api_key = "432f0e6fea8518aa9f06d499d04f62ef"
    city = f"{district_name},{state_name},IN"
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={api_key}"

    response = requests.get(url)
    weather_summary = "Check Weather Report Section"
    if response.status_code == 200:
        data = response.json()
        temps, humidity, precip = [], [], []
        for item in data['list']:
            temps.append(item['main']['temp'])
            humidity.append(item['main']['humidity'])
            if "rain" in item and "3h" in item["rain"]:
                precip.append(item["rain"]["3h"])
            elif "snow" in item and "3h" in item["snow"]:
                precip.append(item["snow"]["3h"])
            else:
                precip.append(0.0)
        weather_summary = f"Avg Temp: {sum(temps)/len(temps):.1f}°C, Avg Humidity: {sum(humidity)/len(humidity):.1f}%, Total Precip: {sum(precip):.1f}mm"

    # Final report dictionary
    report = {
        "Location": f"{district_name}, {state_name}, India",
        "Period": f"{start} to {end}",
        "Vegetation_Indices": indices,
        "Soil_Condition": soil_summary,
        "Crop_Health": crop_health,
        "Pest_Risk": f"{pest_risk} ({pest_risk_percent}%)",
        "Weather_Summary": weather_summary
    }

    return report

# ---------------------- API Endpoint: JSON ----------------------
@app.get("/report")
def get_report(state: str, district: str):
    report = generate_report(state, district)
    return JSONResponse(content=report)

# ---------------------- API Endpoint: PDF Download ----------------------
@app.get("/download_report")
def download_report(state: str, district: str):
    report = generate_report(state, district)

    filename = f"report_{district}_{state}.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("====== Crop Health & Soil Report ======", styles['Title']))
    elements.append(Spacer(1, 12))
    for key, val in report.items():
        elements.append(Paragraph(f"<b>{key}:</b> {val}", styles['Normal']))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return FileResponse(filename, media_type="application/pdf", filename=filename)