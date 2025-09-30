from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import ee
import datetime
import os
import json
import requests
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------- Step 1: Authenticate Earth Engine ----------
serviceaccount = "gee-backend@ee-bairagisayan464.iam.gserviceaccount.com"
key_json = os.getenv("EE_KEY_JSON")  # still using env for security

if key_json is None:
    raise Exception("Please set EE_KEY_JSON in Heroku Config Vars.")

key_dict = json.loads(key_json)

with open("temp_key.json", "w") as f:
    json.dump(key_dict, f)

credentials = ee.ServiceAccountCredentials(serviceaccount, "temp_key.json")
ee.Initialize(credentials, project="ee-bairagisayan464")

# ---------- Step 2: FastAPI App ----------
app = FastAPI(title="Crop Health & Soil Report API")

# ---------- Step 3: Helper Functions ----------
def s2Mask(img):
    qa = img.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return img.updateMask(mask)

def compute_indices(img):
    ndvi = img.normalizedDifference(["B8","B4"]).rename("NDVI")
    evi = img.expression('2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',
                         {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")}).rename("EVI")
    savi = img.expression('((NIR-RED)/((NIR+RED+L)*(1+L)))',
                          {"NIR": img.select("B8"), "RED": img.select("B4"), "L":0.5}).rename("SAVI")
    gndvi = img.normalizedDifference(["B8","B3"]).rename("GNDVI")
    ndwi = img.normalizedDifference(["B3","B8"]).rename("NDWI")
    msi = img.expression("(SWIR/NIR)", {"SWIR": img.select("B11"), "NIR": img.select("B8")}).rename("MSI")
    return {"NDVI": ndvi, "EVI": evi, "SAVI": savi, "GNDVI": gndvi, "NDWI": ndwi, "MSI": msi}

def mean_index(img, name, region):
    val = img.reduceRegion(reducer=ee.Reducer.mean(),
                           geometry=region.geometry(),
                           scale=30,
                           maxPixels=1e13,
                           bestEffort=True).get(name)
    return val.getInfo() if val else None

# ---------- Step 4: OpenWeather API Config ----------
OPENWEATHER_API_KEY = "432f0e6fea8518aa9f06d499d04f62ef"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

# ---------- Step 5: Generate Report ----------
def generate_report(state: str, district: str):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30)

    fc = ee.FeatureCollection("projects/ee-bairagisayan464/assets/GADM-IND")
    region = fc.filter(ee.Filter.And(ee.Filter.eq("NAME_1", state),
                                    ee.Filter.eq("NAME_2", district)))

    if region.size().getInfo() == 0:
        return {"error": "Region not found in GEE assets"}

    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterDate(str(start), str(end))
                  .filterBounds(region)
                  .map(s2Mask))

    if collection.size().getInfo() == 0:
        return {"error": "No Sentinel-2 images found for this location and date range"}

    image = collection.median().clip(region)
    indices_img = compute_indices(image)
    indices = {k: mean_index(v, k, region) for k,v in indices_img.items()}

    # Soil & Crop Health Summary
    soil_summary = ""
    savi_val = indices.get("SAVI")
    msi_val = indices.get("MSI")
    ndwi_val = indices.get("NDWI")
    if savi_val and savi_val < 0.3:
        soil_summary += "Very low vegetation cover (bare soil).\n"
    elif savi_val and savi_val < 0.6:
        soil_summary += "Moderate vegetation cover.\n"
    else:
        soil_summary += "Dense healthy vegetation.\n"
    if msi_val and msi_val > 1.5:
        soil_summary += "High moisture stress detected.\n"
    else:
        soil_summary += "Soil moisture is within healthy range.\n"
    if ndwi_val and ndwi_val < 0:
        soil_summary += "Low water presence, possible dry condition.\n"
    else:
        soil_summary += "Adequate water presence in soil.\n"

    # Crop Health & Pest Risk
    crop_health = "Normal"
    risk_score = 0
    if indices.get("NDVI") and indices["NDVI"] < 0.3:
        risk_score += 30
    if msi_val and msi_val > 1.5:
        risk_score += 25
    if ndwi_val and ndwi_val < 0.2:
        risk_score += 15
    pest_risk_percent = min(100, risk_score)
    pest_risk = "Low" if pest_risk_percent < 30 else "Moderate" if pest_risk_percent<60 else "High"
    if indices.get("NDVI") and indices["NDVI"] > 0.6 and ndwi_val and ndwi_val > 0:
        crop_health = "Healthy"
    elif indices.get("NDVI") and indices["NDVI"] < 0.3:
        crop_health = "Poor / Stress detected"

    # Weather
    city = f"{district},{state},IN"
    url = f"{OPENWEATHER_BASE_URL}?q={city}&units=metric&appid={OPENWEATHER_API_KEY}"
    weather_summary = "Check Weather Report Section"
    resp = requests.get(url)
    if resp.status_code==200:
        data = resp.json()
        temps, humidity, precip = [],[],[]
        for item in data["list"]:
            temps.append(item["main"]["temp"])
            humidity.append(item["main"]["humidity"])
            if "rain" in item and "3h" in item["rain"]:
                precip.append(item["rain"]["3h"])
            else:
                precip.append(0.0)
        weather_summary = f"Avg Temp: {sum(temps)/len(temps):.1f}°C, Avg Humidity: {sum(humidity)/len(humidity):.1f}%, Total Precip: {sum(precip):.1f}mm"

    report = {
        "Location": f"{district}, {state}, India",
        "Period": f"{start} to {end}",
        "Vegetation_Indices": indices,
        "Soil_Condition": soil_summary,
        "Crop_Health": crop_health,
        "Pest_Risk": f"{pest_risk} ({pest_risk_percent}%)",
        "Weather_Summary": weather_summary
    }
    return report

# ---------- Step 6: API Endpoints ----------
@app.get("/report")
def report_json(state: str = Query(...), district: str = Query(...)):
    report = generate_report(state, district)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return JSONResponse(report)

@app.get("/download_report")
def download_report(state: str = Query(...), district: str = Query(...)):
    report = generate_report(state, district)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])

    # Generate PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("====== Crop Health & Soil Report ======", styles['Title']))
    elements.append(Spacer(1,12))
    for key, val in report.items():
        elements.append(Paragraph(f"<b>{key}:</b> {val}", styles['Normal']))
        elements.append(Spacer(1,12))

    doc.build(elements)
    buffer.seek(0)

    return FileResponse(buffer, media_type="application/pdf",
                        filename=f"report_{district}_{state}.pdf")
