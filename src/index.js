// Weather-app Worker: static assets + /api/judge (AI verdict via OpenRouter)

const SYSTEM_PROMPT = `You are a concise weather analyst. You receive weather forecasts from multiple independent models for a specific location and must produce a single JSON judgment.

Return ONLY a JSON object with these fields:
{
  "verdict": string  // 3-8 word summary, e.g. "Clear and warm, high confidence"
  "confidence": "high" | "medium" | "low"
  "reasoning": string  // 1-2 sentences explaining WHY. Reference specific model disagreements or agreements. Mention the source(s) you trust most and why.
  "final_temp_c": number  // your best-guess single temperature in Celsius
  "rain_next_2h": boolean  // will it rain in the next 2 hours?
  "advice": string  // one short sentence of practical advice for the user
}

Rules:
- Do NOT just average. Weight ECMWF highest (best global model), then ICON, then GFS, then GEM, then MetNo (unless location is Nordic, then boost MetNo).
- If models diverge >3°C, confidence is "low" and say why (front, storm, complex terrain?).
- If they agree within 1°C, confidence is "high".
- Be honest about uncertainty. Better to say "models disagree" than to fake a precise number.
- Respond with ONLY the JSON, no markdown fences, no prose.`;

async function handleJudge(request, env) {
  if (request.method !== "POST") {
    return json({ error: "POST only" }, 405);
  }
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "invalid JSON" }, 400);
  }

  const { lat, lon, models } = body;
  if (!Array.isArray(models) || models.length === 0) {
    return json({ error: "missing models[]" }, 400);
  }
  if (!env.OPENROUTER_API_KEY) {
    return json({ error: "server misconfigured: no API key" }, 500);
  }

  const userPrompt = `Location: lat=${lat}, lon=${lon}
Current time: ${new Date().toISOString()}

Model readings (current conditions):
${models.map(m => `- ${m.name}: ${m.temp_c}°C, weather_code=${m.weather_code}${m.precip_prob_1h != null ? `, precip_prob_next_1h=${m.precip_prob_1h}%` : ""}`).join("\n")}

Weather codes: 0=clear, 1-3=partly cloudy to overcast, 45-48=fog, 51-55=drizzle, 61-65=rain, 71-75=snow, 80-82=showers, 95-99=thunderstorm.

Judge this ensemble and return your verdict JSON.`;

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
        { role: "user", content: userPrompt }
      ],
      temperature: 0.2,
      max_tokens: 400,
      response_format: { type: "json_object" }
    })
  });

  if (!resp.ok) {
    const text = await resp.text();
    return json({ error: "openrouter failed", detail: text.slice(0, 500) }, 502);
  }

  const data = await resp.json();
  const content = data?.choices?.[0]?.message?.content;
  if (!content) return json({ error: "empty AI response" }, 502);

  let verdict;
  try {
    verdict = JSON.parse(content);
  } catch {
    return json({ error: "AI returned non-JSON", raw: content.slice(0, 500) }, 502);
  }

  return json(verdict, 200, {
    "Cache-Control": "public, max-age=300"
  });
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
    if (url.pathname === "/api/judge") {
      return handleJudge(request, env);
    }
    return env.ASSETS.fetch(request);
  }
};
