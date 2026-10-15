type Identity = { userId: string } | null;

export async function forwardAgentRequest(request: Request, path: string[], identity: Identity,
  settings: { base?: string; token?: string }, upstream: typeof fetch = fetch): Promise<Response> {
  if (!identity) return Response.json({ error: "sign_in_required" }, { status: 401 });
  const joined = path.join("/");
  if (!/^(credentials(\/delete)?|credential-sessions|conversations|conversations\/[a-f0-9-]{36}|conversations\/[a-f0-9-]{36}\/(uploads|runs)|runs\/[a-f0-9-]{36}(\/cancel)?)$/.test(joined))
    return Response.json({ error: "not_found" }, { status: 404 });
  if (!settings.base || !settings.token) return Response.json({ error: "service_not_configured" }, { status: 503 });
  if (request.method !== "GET" && request.headers.get("origin") !== new URL(request.url).origin)
    return Response.json({ error: "invalid_origin" }, { status: 403 });
  if (Number(request.headers.get("content-length") || 0) > 33 * 1024 * 1024)
    return Response.json({ error: "file_too_large" }, { status: 413 });
  const headers = new Headers({ authorization: `Bearer ${settings.token}`, "x-agent-user": identity.userId });
  for (const key of ["content-type", "idempotency-key"])
    if (request.headers.has(key)) headers.set(key, request.headers.get(key)!);
  if (request.method === "POST" && path[0] === "conversations" && path.length === 3 && path[2] === "runs" && request.headers.has("x-agent-credential-handle"))
    headers.set("x-agent-credential-handle", request.headers.get("x-agent-credential-handle")!);
  const offset = new URL(request.url).searchParams.get("offset");
  const query = request.method === "GET" && path[0] === "conversations" && path.length === 2 && offset && /^\d{1,9}$/.test(offset) ? `?offset=${offset}` : "";
  try {
    const response = await upstream(`${settings.base.replace(/\/$/, "")}/v1/${joined}${query}`, {
      method: request.method, headers, body: request.method === "GET" ? undefined : request.body,
      redirect: "error", signal: AbortSignal.timeout(60_000),
      ...({ duplex: "half" } as Record<string, string>),
    });
    return new Response(response.body, { status: response.status,
      headers: { "content-type": "application/json", "cache-control": "no-store" } });
  } catch {
    return Response.json({ error: "service_unreachable" }, { status: 503 });
  }
}

export function safeSourceUrl(value: string): string | null {
  try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) ? url.href : null; }
  catch { return null; }
}
