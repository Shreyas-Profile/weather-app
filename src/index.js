// Weather app Worker: sensor triangulation + satellite + AI in the loop
//
// Data sources (all free, no API key needed except OpenRouter for the AI):
//   1. METAR — real airport weather stations from aviationweather.gov
//      (temperature, dewpoint, wind, cloud cover, visibility, pressure)
//   2. Satellite/radar-derived current conditions via Open-Meteo
//      (cloud_cover %, precipitation mm, rain, snowfall, weather_code)
//   3. Multi-model forecast ensemble (GFS + ECMWF + ICON) for future hours
//   4. OpenRouter → Gemini judges everything and returns a final call

const SYSTEM_PROMPT = `You are a meteorologist. You get:
- Real airport weather station readings (METAR) at known distances from the user
- Satellite/radar-derived current sky and precipitation state
- Multi-model forecast for temperature

Your job: produce ONE JSON judgment for the user's exact spot.

Return ONLY this JSON (no fences, no prose):
{
  "temp_c": number,           // final temperature in Celsius
  "condition": "sunny" | "partly_cloudy" | "cloudy" | "overcast" | "rain" | "snow" | "thunderstorm" | "fog",
  "confidence": "high" | "medium" | "low",
  "reasoning": string,        // 1-2 sentences: which sources you trusted most and why
  "sources_used": string[],   // e.g. ["EGLL (Heathrow) 6.2km", "Satellite cloud cover", "ECMWF model"]
  "advice": string            // one short practical sentence
}

Rules:
- Weight sensor readings by INVERSE distance squared. A station 2km away counts 4x more than one 4km away.
- Correct for ELEVATION: temp drops ~6.5°C per 1000m elevation gain. If a sensor is 500m higher than the user, its reading is ~3°C colder than the user experiences.
- For rain/snow/cloud state, trust satellite/radar over forecast models. Sensors can be blocked by trees, so trust them for temperature more than for sky state.
- If sensors are all >20km away, confidence drops.
- If sensors disagree with satellite (e.g., station says clear but satellite says overcast), say so in reasoning and lean satellite for sky, sensor for temp.
- Spread rule for confidence: if sensor temps disagree by <1°C → high; 1-3°C → medium; >3°C → low.`;

// Haversine distance in km
function distKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const toRad = d => d * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// Fetch nearest METAR stations within a bounding box that expands until we have enough
async function fetchNearbyMetars(lat, lon) {
  for (const halfDeg of [0.5, 1.0, 2.0, 4.0]) {  // ~55, 110, 220, 440 km half-widths
    const bbox = `${lat - halfDeg},${lon - halfDeg},${lat + halfDeg},${lon + halfDeg}`;
    const r = await fetch(`https://aviationweather.gov/api/data/metar?bbox=${bbox}&format=json`);
    if (!r.ok) continue;
    const arr = await r.json();
    if (!Array.isArray(arr) || arr.length === 0) continue;
    const withDist = arr
      .filter(m => m.temp != null && typeof m.lat === "number")
      .map(m => ({ ...m, distance_km: distKm(lat, lon, m.lat, m.lon) }))
      .sort((a, b) => a.distance_km - b.distance_km)
      .slice(0, 5);
    if (withDist.length > 0) return withDist;
  }
  return [];
}

async function fetchCurrentSatellite(lat, lon) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
              `&current=temperature_2m,cloud_cover,precipitation,rain,snowfall,weather_code,wind_speed_10m` +
              `&hourly=temperature_2m,precipitation_probability,cloud_cover` +
              `&forecast_days=1&timezone=auto`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Open-Meteo failed: ${r.status}`);
  return r.json();
}

async function fetchModelEnsemble(lat, lon) {
  const models = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"];
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
              `&current=temperature_2m,weather_code&models=${models.join(",")}&timezone=auto`;
  const r = await fetch(url);
  if (!r.ok) return null;
  const data = await r.json();
  const out = [];
  for (const m of models) {
    const t = data.current?.[`temperature_2m_${m}`];
    const c = data.current?.[`weather_code_${m}`];
    if (t != null) out.push({ model: m, temp_c: t, weather_code: c });
  }
  return out;
}

async function askAI(env, payload) {
  const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://weather.shreyas.uk",
      "X-Title": "Ensemble Weather"
    },
    body: JSON.stringify({
      model: "google/gemini-2.5-flash-lite",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: JSON.stringify(payload, null, 2) }
      ],
      temperature: 0.1,
      max_tokens: 600,
      response_format: { type: "json_object" }
    })
  });
  if (!resp.ok) throw new Error(`OpenRouter ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  const content = data?.choices?.[0]?.message?.content;
  if (!content) throw new Error("Empty AI response");
  return JSON.parse(content);
}

async function handleWeather(request, env) {
  const url = new URL(request.url);
  const lat = parseFloat(url.searchParams.get("lat"));
  const lon = parseFloat(url.searchParams.get("lon"));
  if (isNaN(lat) || isNaN(lon)) {
    return json({ error: "missing or invalid lat/lon" }, 400);
  }
  if (!env.OPENROUTER_API_KEY) {
    return json({ error: "server misconfigured: no AI key" }, 500);
  }

  const [metars, satellite, models] = await Promise.all([
    fetchNearbyMetars(lat, lon).catch(e => ({ error: e.message })),
    fetchCurrentSatellite(lat, lon).catch(e => ({ error: e.message })),
    fetchModelEnsemble(lat, lon).catch(e => null)
  ]);

  const sensorReadings = Array.isArray(metars) ? metars.map(m => ({
    name: m.name || m.icaoId,
    icao: m.icaoId,
    distance_km: +m.distance_km.toFixed(1),
    elevation_m: m.elev,
    temp_c: m.temp,
    dewpoint_c: m.dewp,
    wind_kt: m.wspd,
    cloud_cover: m.cover,      // CLR, FEW, SCT, BKN, OVC, CAVOK
    visibility: m.visib,
    observed_at: m.reportTime
  })) : [];

  const sat = satellite.current ? {
    temperature_c: satellite.current.temperature_2m,
    cloud_cover_pct: satellite.current.cloud_cover,
    precipitation_mm: satellite.current.precipitation,
    rain_mm: satellite.current.rain,
    snowfall_cm: satellite.current.snowfall,
    weather_code: satellite.current.weather_code,
    wind_speed_kmh: satellite.current.wind_speed_10m,
    time: satellite.current.time,
    elevation_m: satellite.elevation
  } : null;

  const payload = {
    user_location: { lat, lon, elevation_m: sat?.elevation_m ?? null },
    airport_sensors_nearby: sensorReadings,
    satellite_radar_current: sat,
    forecast_models_current: models || [],
    wmo_code_meanings: "0=clear, 1-3=partly to overcast, 45-48=fog, 51-55=drizzle, 61-65=rain, 71-75=snow, 80-82=showers, 95-99=thunderstorm"
  };

  let verdict;
  try {
    verdict = await askAI(env, payload);
  } catch (e) {
    return json({ error: "AI call failed", detail: e.message, sources: { sensors: sensorReadings, satellite: sat } }, 502);
  }

  return json({
    verdict,
    sources: {
      sensors: sensorReadings,
      satellite: sat,
      models: models || []
    }
  }, 200, { "Cache-Control": "public, max-age=180" });
}

function json(obj, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      ...extraHeaders
    }
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/weather") return handleWeather(request, env);
    return env.ASSETS.fetch(request);
  }
};
