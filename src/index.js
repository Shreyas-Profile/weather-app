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
- **Prefer Netatmo neighborhood stations over airport METAR when available** — they're closer (usually <5km vs 15km+) but be aware they're personal weather stations: some are sited badly (sun exposure, roofs, walls) and may read 2-5°C too hot. If Netatmo readings cluster tightly (spread < 2°C), trust the cluster. If one Netatmo is way hotter than nearby ones, treat it as an outlier (bad sensor placement).
- Weight sensor readings by INVERSE distance squared. A station 1km away counts 100x more than one 10km away.
- Correct for ELEVATION: temp drops ~6.5°C per 1000m elevation gain. If a sensor is 500m higher than the user, its reading is ~3°C colder than the user experiences.
- For rain/snow/cloud state, trust satellite/radar over forecast models. Sensors can be blocked by trees, so trust them for temperature more than for sky state.
- **CRITICAL for the "condition" field:** decide raining vs not-raining from \`nowcast_minutely\` (radar-blended, minute-resolution). If \`nowcast_minutely.raining_right_now\` is FALSE, condition MUST NOT be "rain"/"snow"/"thunderstorm" — use "cloudy"/"overcast"/"partly_cloudy" based on cloud_cover_pct. The satellite \`current.weather_code\` and \`current.precipitation\` fields can be stale (model output that lags real radar) — trust nowcast over them.
- If ALL sensors are >20km away, confidence drops.
- If sensors disagree with satellite (e.g., station says clear but satellite says overcast), say so in reasoning and lean satellite for sky, sensor for temp.
- Spread rule for confidence: if the trusted sensor cluster agrees within <1°C → high; 1-3°C → medium; >3°C → low.`;

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

// Cache a fresh access token in module scope so we don't refresh on every request.
// Access tokens live ~3 hours; we refresh a bit early to be safe.
let netatmoAccess = { token: null, expiresAt: 0 };

async function getNetatmoAccessToken(env) {
  if (netatmoAccess.token && Date.now() < netatmoAccess.expiresAt - 60_000) {
    return netatmoAccess.token;
  }
  const r = await fetch("https://api.netatmo.com/oauth2/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: env.NETATMO_REFRESH_TOKEN,
      client_id: env.NETATMO_CLIENT_ID,
      client_secret: env.NETATMO_CLIENT_SECRET
    })
  });
  if (!r.ok) throw new Error(`Netatmo token refresh ${r.status}: ${await r.text()}`);
  const data = await r.json();
  netatmoAccess = {
    token: data.access_token,
    expiresAt: Date.now() + (data.expires_in || 10800) * 1000
  };
  return data.access_token;
}

async function fetchNetatmoStations(lat, lon, env) {
  if (!env.NETATMO_CLIENT_ID || !env.NETATMO_REFRESH_TOKEN) return [];
  const halfDeg = 0.1;  // ~11km bbox
  try {
    const token = await getNetatmoAccessToken(env);
    const url = `https://api.netatmo.com/api/getpublicdata` +
                `?lat_ne=${lat + halfDeg}&lon_ne=${lon + halfDeg}` +
                `&lat_sw=${lat - halfDeg}&lon_sw=${lon - halfDeg}` +
                `&required_data=temperature&filter=true`;
    const r = await fetch(url, { headers: { "Authorization": `Bearer ${token}` } });
    if (!r.ok) return [];
    const data = await r.json();
    if (!Array.isArray(data.body)) return [];
    const now = Math.floor(Date.now() / 1000);
    const stations = [];
    for (const s of data.body) {
      const [sLon, sLat] = s.place.location;
      // Pull the freshest temperature/humidity measurement out of the tangled 'measures' object.
      let temp = null, humidity = null, timestamp = null;
      for (const mod of Object.values(s.measures || {})) {
        if (mod.type?.includes("temperature") && mod.res) {
          const [ts, vals] = Object.entries(mod.res)[0] || [];
          if (ts && vals) {
            const tIdx = mod.type.indexOf("temperature");
            const hIdx = mod.type.indexOf("humidity");
            temp = vals[tIdx];
            humidity = hIdx >= 0 ? vals[hIdx] : null;
            timestamp = parseInt(ts);
          }
        }
      }
      if (temp == null || timestamp == null) continue;
      if (now - timestamp > 3600) continue;  // skip readings older than 1 hour
      stations.push({
        name: `${s.place.street || "unnamed"}, ${s.place.city || ""}`,
        distance_km: distKm(lat, lon, sLat, sLon),
        elevation_m: s.place.altitude,
        temp_c: temp,
        humidity_pct: humidity,
        observed_at: new Date(timestamp * 1000).toISOString()
      });
    }
    stations.sort((a, b) => a.distance_km - b.distance_km);
    return stations.slice(0, 8).map(s => ({ ...s, distance_km: +s.distance_km.toFixed(2) }));
  } catch (e) {
    return [];
  }
}

async function fetchCurrentSatellite(lat, lon) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
              `&current=temperature_2m,apparent_temperature,cloud_cover,precipitation,rain,snowfall,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,relative_humidity_2m,pressure_msl,uv_index,visibility` +
              `&minutely_15=precipitation,rain,snowfall,weather_code` +
              `&hourly=temperature_2m,apparent_temperature,precipitation,precipitation_probability,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,uv_index` +
              `&daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,wind_direction_10m_dominant,uv_index_max` +
              `&forecast_days=7&forecast_minutely_15=48&timezone=auto`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Open-Meteo failed: ${r.status}`);
  return r.json();
}

async function handleGeocode(request) {
  const url = new URL(request.url);
  const q = (url.searchParams.get("q") || "").trim();
  if (q.length < 2) return json({ results: [] });
  try {
    const r = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q)}&count=8&language=en&format=json`);
    if (!r.ok) return json({ results: [] });
    const data = await r.json();
    const results = (data.results || []).map(p => ({
      name: p.name,
      admin1: p.admin1,
      country: p.country,
      country_code: p.country_code,
      lat: p.latitude,
      lon: p.longitude,
      elevation: p.elevation
    }));
    return json({ results }, 200, { "Cache-Control": "public, max-age=86400" });
  } catch (e) {
    return json({ results: [], error: e.message });
  }
}

function intensityLabel(mmPerHour, weatherCode) {
  if (weatherCode >= 71 && weatherCode <= 77) return "snow";
  if (weatherCode === 85 || weatherCode === 86) return "snow showers";
  if (weatherCode === 66 || weatherCode === 67) return "sleet";
  if (weatherCode >= 95) return "thunderstorm";
  if (mmPerHour <= 0.05) return null;
  if (mmPerHour < 0.5) return "drizzle";
  if (mmPerHour < 2.5) return "light rain";
  if (mmPerHour < 7.6) return "rain";
  if (mmPerHour < 50) return "heavy rain";
  return "torrential rain";
}

/**
 * Turn Open-Meteo minutely_15 (four bins per hour) into a minute-precision
 * nowcast: per-minute mm/h array (120 min), compressed events, human summary.
 * Each 15-min bucket's mm total is treated as constant across its minutes:
 * mm/h == mm_in_15_min * 4 for every minute inside that bucket.
 */
function buildNowcast(minutely) {
  if (!minutely || !minutely.time || !minutely.precipitation) return null;
  const now = Date.now();
  let startIdx = minutely.time.findIndex(t => new Date(t).getTime() > now) - 1;
  if (startIdx < 0) startIdx = 0;

  const perMinute = [];
  for (let m = 0; m < 120; m++) {
    const targetMs = now + m * 60000;
    let slot = startIdx;
    for (let i = startIdx; i < minutely.time.length; i++) {
      if (new Date(minutely.time[i]).getTime() <= targetMs) slot = i;
      else break;
    }
    const mm15 = minutely.precipitation[slot] || 0;
    const mmH = mm15 * 4;
    const wc = minutely.weather_code?.[slot];
    perMinute.push({ minute: m, mm_per_h: +mmH.toFixed(2), weather_code: wc });
  }

  const events = [];
  let inEv = false, evStart = 0, evPeak = 0, evPeakWC = null;
  for (let i = 0; i < perMinute.length; i++) {
    const p = perMinute[i];
    const raining = p.mm_per_h > 0.05;
    if (raining && !inEv) { inEv = true; evStart = i; evPeak = p.mm_per_h; evPeakWC = p.weather_code; }
    else if (raining) { if (p.mm_per_h > evPeak) { evPeak = p.mm_per_h; evPeakWC = p.weather_code; } }
    else if (inEv) {
      inEv = false;
      events.push({ start_min: evStart, end_min: i - 1, peak_mm_h: +evPeak.toFixed(2), type: intensityLabel(evPeak, evPeakWC) });
    }
  }
  if (inEv) events.push({ start_min: evStart, end_min: 119, peak_mm_h: +evPeak.toFixed(2), type: intensityLabel(evPeak, evPeakWC), continues: true });

  let summary;
  if (events.length === 0) summary = "No precipitation expected in the next 2 hours";
  else {
    const parts = [];
    events.slice(0, 3).forEach((ev, idx) => {
      if (idx === 0) {
        if (ev.start_min === 0) parts.push(`${ev.type} now${ev.end_min < 119 ? `, stopping in ${ev.end_min + 1} min` : ""}`);
        else parts.push(`${ev.type} starting in ${ev.start_min} min${ev.end_min < 119 ? `, ending in ${ev.end_min + 1} min` : ""}`);
      } else {
        parts.push(`then ${ev.type} from ${ev.start_min} min`);
      }
    });
    summary = parts.join(", ");
  }

  return { summary, events, per_minute: perMinute, raining_right_now: perMinute[0].mm_per_h > 0.05 };
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

  const [metars, satellite, models, netatmo] = await Promise.all([
    fetchNearbyMetars(lat, lon).catch(e => ({ error: e.message })),
    fetchCurrentSatellite(lat, lon).catch(e => ({ error: e.message })),
    fetchModelEnsemble(lat, lon).catch(e => null),
    fetchNetatmoStations(lat, lon, env).catch(e => [])
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
    feels_like_c: satellite.current.apparent_temperature,
    cloud_cover_pct: satellite.current.cloud_cover,
    precipitation_mm: satellite.current.precipitation,
    rain_mm: satellite.current.rain,
    snowfall_cm: satellite.current.snowfall,
    weather_code: satellite.current.weather_code,
    wind_speed_kmh: satellite.current.wind_speed_10m,
    wind_direction_deg: satellite.current.wind_direction_10m,
    wind_gusts_kmh: satellite.current.wind_gusts_10m,
    humidity_pct: satellite.current.relative_humidity_2m,
    pressure_hpa: satellite.current.pressure_msl,
    uv_index: satellite.current.uv_index,
    visibility_m: satellite.current.visibility,
    time: satellite.current.time,
    elevation_m: satellite.elevation
  } : null;

  // Build minute-precision rain nowcast from Open-Meteo minutely_15
  const nowcast = buildNowcast(satellite.minutely_15);

  // Extract next-rain info from hourly probability
  const hourly = satellite.hourly || {};
  const nowIdx = (hourly.time || []).findIndex(t => new Date(t).getTime() >= Date.now());
  const startIdx = Math.max(0, nowIdx);
  let nextRainHours = null;
  if (hourly.precipitation_probability) {
    for (let i = startIdx; i < Math.min(startIdx + 24, hourly.precipitation_probability.length); i++) {
      if (hourly.precipitation_probability[i] >= 50) {
        nextRainHours = i - startIdx;
        break;
      }
    }
  }

  // Return the FULL 7-day hourly (168h) so the UI can filter per day when a day is clicked
  const hourlyForecast = [];
  for (let i = 0; i < (hourly.time || []).length; i++) {
    hourlyForecast.push({
      time: hourly.time[i],
      temp_c: hourly.temperature_2m?.[i],
      feels_like_c: hourly.apparent_temperature?.[i],
      weather_code: hourly.weather_code?.[i],
      precip_mm: hourly.precipitation?.[i],
      precip_prob_pct: hourly.precipitation_probability?.[i],
      cloud_cover_pct: hourly.cloud_cover?.[i],
      wind_speed_kmh: hourly.wind_speed_10m?.[i],
      wind_direction_deg: hourly.wind_direction_10m?.[i],
      uv_index: hourly.uv_index?.[i]
    });
  }

  const dailyRaw = satellite.daily || {};
  const daily = (dailyRaw.time || []).map((t, i) => ({
    date: t,
    weather_code: dailyRaw.weather_code?.[i],
    temp_max_c: dailyRaw.temperature_2m_max?.[i],
    temp_min_c: dailyRaw.temperature_2m_min?.[i],
    feels_max_c: dailyRaw.apparent_temperature_max?.[i],
    feels_min_c: dailyRaw.apparent_temperature_min?.[i],
    sunrise: dailyRaw.sunrise?.[i],
    sunset: dailyRaw.sunset?.[i],
    precip_sum_mm: dailyRaw.precipitation_sum?.[i],
    precip_prob_max_pct: dailyRaw.precipitation_probability_max?.[i],
    wind_max_kmh: dailyRaw.wind_speed_10m_max?.[i],
    wind_dominant_deg: dailyRaw.wind_direction_10m_dominant?.[i],
    uv_max: dailyRaw.uv_index_max?.[i]
  }));

  const payload = {
    user_location: { lat, lon, elevation_m: sat?.elevation_m ?? null },
    neighborhood_stations_nearby: netatmo || [],  // Netatmo garden weather stations (best density, but PWS quality varies)
    airport_sensors_nearby: sensorReadings,        // METAR — quality-controlled, but sparse
    satellite_radar_current: sat,
    nowcast_minutely: nowcast ? {
      raining_right_now: nowcast.raining_right_now,
      summary: nowcast.summary,
      events: nowcast.events
    } : null,
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
    forecast: {
      hourly: hourlyForecast,
      daily: daily,
      next_rain_hours: nextRainHours,
      nowcast: nowcast
    },
    sources: {
      netatmo: netatmo || [],
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
    if (url.pathname === "/api/geocode") return handleGeocode(request);
    return env.ASSETS.fetch(request);
  }
};
