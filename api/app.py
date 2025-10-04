# app.py
import os, json
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import httpx

from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# OpenAI SDK (>=1.0.0)
import openai

# ---------- Carga de variables ----------
load_dotenv()

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ---------- App y CORS ----------
app = FastAPI(title="MyCity Dashboard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Base de datos (incidencias) ----------
engine = create_engine("sqlite:///./data.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class IncidentORM(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(20), index=True)
    lat = Column(Float, index=True)
    lng = Column(Float, index=True)
    description = Column(Text)
    reporter = Column(String(80))

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Esquemas ----------
class LatLng(BaseModel):
    lat: float
    lng: float

class TrafficQuery(BaseModel):
    # bbox: [minLon, minLat, maxLon, maxLat]
    bbox: List[float]
    provider: Literal["tomtom"] = "tomtom"

class PollutionQuery(BaseModel):
    coords: LatLng
    source: Literal["openweather"] = "openweather"

class IncidentCreate(BaseModel):
    type: Literal["basura","robo","bache","accidente","otro"] = "otro"
    location: LatLng
    description: Optional[str] = None
    reporter: Optional[str] = None

class Incident(IncidentCreate):
    id: int = Field(...)

class AIInsightRequest(BaseModel):
    city: str = "Riobamba"
    metrics: dict  # e.g., {"pm25":22, "traffic_index":0.58, "incidents_24h":9}

# ---------- Health ----------
@app.get("/health")
async def health():
    return {"ok": True}

# ---------- Tráfico: TomTom ----------
@app.post("/traffic")
async def traffic(q: TrafficQuery):
    if not TOMTOM_API_KEY:
        raise HTTPException(500, "Falta TOMTOM_API_KEY")

    # TomTom Flow Segment Data usa un punto + zoom; tomamos centro del bbox
    lon = (q.bbox[0] + q.bbox[2]) / 2
    lat = (q.bbox[1] + q.bbox[3]) / 2

    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {"point": f"{lat},{lon}", "key": TOMTOM_API_KEY}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    # respuesta directa + normalización mínima
    return {"provider": "tomtom", "data": data}

# ---------- Contaminación: OpenWeather Air Pollution ----------
@app.post("/pollution")
async def pollution(q: PollutionQuery):
    if not OPENWEATHER_API_KEY:
        raise HTTPException(500, "Falta OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": q.coords.lat, "lon": q.coords.lng, "appid": OPENWEATHER_API_KEY}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    return {"source": "openweather", "data": data}

# ---------- Incidencias: CRUD mínimo ----------
@app.post("/incidents", response_model=Incident)
def create_incident(b: IncidentCreate, db: Session = Depends(get_db)):
    m = IncidentORM(
        type=b.type,
        lat=b.location.lat,
        lng=b.location.lng,
        description=b.description,
        reporter=b.reporter,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return Incident(id=m.id, **b.model_dump())

@app.get("/incidents", response_model=List[Incident])
def list_incidents(db: Session = Depends(get_db)):
    items = db.query(IncidentORM).order_by(IncidentORM.id.desc()).limit(200).all()
    return [
        Incident(
            id=i.id,
            type=i.type,
            location={"lat": i.lat, "lng": i.lng},
            description=i.description,
            reporter=i.reporter,
        )
        for i in items
    ]

# ---------- IA asistente (OpenAI) ----------
@app.post("/ai/insights")
async def ai_insights(req: AIInsightRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Falta OPENAI_API_KEY")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        f"Genera un brief con 3 acciones priorizadas para {req.city} "
        f"usando estos datos urbanos actuales: {json.dumps(req.metrics)}. "
        f"Sé concreto y operativo."
    )

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "Eres un analista urbano conciso y práctico."},
            {"role": "user", "content": prompt},
        ],
    )
    text = chat.choices[0].message.content
    return {"insights": text}

# ---------- Satélite: plantilla WMTS de NASA GIBS ----------
@app.get("/tiles/gibs")
def gibs_tiles(
    layer: str = "MODIS_Terra_CorrectedReflectance_TrueColor",
    time: str = "2025-01-01",
):
    # El frontend usará este template XYZ
    template = (
        f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}"
        f"/default/{time}/GoogleMapsCompatible/{{z}}/{{y}}/{{x}}.jpg"
    )
    return {"layer": layer, "template": template}
