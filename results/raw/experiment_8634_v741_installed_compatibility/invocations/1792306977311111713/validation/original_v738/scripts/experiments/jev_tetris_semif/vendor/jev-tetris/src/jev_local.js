// Local Jev-compatible client. It accepts no credentials and permits only loopback.

const ENDPOINT = () => process.env.JEV_LOCAL_ENDPOINT || 'http://127.0.0.1:8940/ask'

function localEndpoint() {
  const endpoint = new URL(ENDPOINT())
  const host = endpoint.hostname.toLowerCase()
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(host)) {
    throw new Error(`JEV_LOCAL_ENDPOINT must use loopback, got ${endpoint.hostname}`)
  }
  if (!['http:', 'https:'].includes(endpoint.protocol)) {
    throw new Error(`JEV_LOCAL_ENDPOINT must use HTTP, got ${endpoint.protocol}`)
  }
  return endpoint
}

export async function ask({ state, questions, key: _key, signal }) {
  const endpoint = localEndpoint()
  const t0 = performance.now()
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ state, questions }),
    signal,
  })
  const ms = Math.round(performance.now() - t0)
  const text = await res.text()
  if (!res.ok) throw new Error(`Local readout ${res.status}: ${text.slice(0, 500)}`)
  let json
  try { json = JSON.parse(text) } catch { throw new Error(`Local readout returned non-JSON: ${text.slice(0, 300)}`) }
  return { raw: json, ms }
}

export function readChoice(raw, name) {
  const a = raw?.answers?.[name] ?? raw?.[name]
  if (!a) return null
  const picked = a.choice ?? a.value ?? a.answer ?? a.key
  const probabilities = a.probabilities ?? a.probs ?? a.distribution ?? {}
  return { picked, probabilities, confidence: a.confidence ?? null }
}

export function readScore(raw, name) {
  const a = raw?.answers?.[name] ?? raw?.[name]
  if (!a) return null
  return { score: a.score ?? a.value ?? null, confidence: a.confidence ?? null, legend: a.legend ?? null }
}

export function readNoul(raw, name) {
  const a = raw?.answers?.[name] ?? raw?.[name]
  if (!a) return null
  return { p: a.noul ?? a.probability ?? a.value ?? null, confidence: a.confidence ?? null }
}
