import test from "node:test";
import assert from "node:assert/strict";
import { forwardAgentRequest, safeSourceUrl } from "../app/lib/proxy.ts";

test("anonymous execution and traversal are rejected", async () => {
  const req = new Request("https://site.example/api/agent/conversations");
  assert.equal((await forwardAgentRequest(req, ["conversations"], null, {})).status, 401);
  assert.equal((await forwardAgentRequest(req, ["..", "health"], {userId:"alice"}, {})).status, 404);
});
test("proxy replaces browser identity and authorization", async () => {
  let seen: Headers | null = null;
  const req = new Request("https://site.example/api/agent/conversations", {
    headers:{"x-agent-user":"bob", authorization:"Bearer attacker"}});
  const result = await forwardAgentRequest(req, ["conversations"], {userId:"alice"},
    {base:"https://backend.example",token:"service-secret"}, async (_url, options) => {
      seen = new Headers(options?.headers); return Response.json([]);
    });
  assert.equal(result.status, 200);
  assert.equal(seen!.get("x-agent-user"), "alice");
  assert.equal(seen!.get("authorization"), "Bearer service-secret");
});
test("cross-origin paid requests and unsafe sources are blocked", async () => {
  const req = new Request("https://site.example/api/agent/conversations", {method:"POST",headers:{origin:"https://evil.example"}});
  assert.equal((await forwardAgentRequest(req,["conversations"],{userId:"alice"},{base:"https://backend.example",token:"secret"})).status,403);
  assert.equal(safeSourceUrl("javascript:alert(1)"), null);
  assert.equal(safeSourceUrl("https://example.org"), "https://example.org/");
});
test("conversation pagination forwards only a validated offset", async () => {
  let seen = "";
  const req = new Request("https://site.example/api/agent/conversations/12345678-1234-1234-1234-123456789abc?offset=100&token=evil");
  await forwardAgentRequest(req,["conversations","12345678-1234-1234-1234-123456789abc"],{userId:"alice"},
    {base:"https://backend.example",token:"secret"},async url => { seen=String(url); return Response.json([]); });
  assert.equal(seen,"https://backend.example/v1/conversations/12345678-1234-1234-1234-123456789abc?offset=100");
});
test("credential routes are protected and handles forwarded only to run starts", async () => {
  for (const path of [["credentials"],["credentials","delete"],["credential-sessions"],
    ["conversations","12345678-1234-1234-1234-123456789abc","runs"]]) {
    let headers = new Headers();
    const result = await forwardAgentRequest(new Request("https://site.example/api/agent/"+path.join("/"), {
      method:"POST",headers:{origin:"https://site.example","x-agent-credential-handle":"fake-handle"}}),path,{userId:"alice"},
      {base:"https://backend.example",token:"secret"},async (_url, options) => {headers=new Headers(options?.headers); return Response.json({});});
    assert.equal(result.status,200);
    assert.equal(headers.get("x-agent-credential-handle"),path.at(-1)==="runs" ? "fake-handle":null);
  }
  const hostile = new Request("https://site.example/api/agent/credentials",{method:"POST",headers:{origin:"https://evil.example"}});
  assert.equal((await forwardAgentRequest(hostile,["credentials"],{userId:"alice"},{base:"https://backend.example",token:"secret"})).status,403);
});
test("Worker-compatible manual redirects never forward credentials to another destination",async()=>{
  let calls=0;
  let redirect:RequestRedirect|undefined;
  const response=await forwardAgentRequest(new Request("https://site.example/api/agent/credentials"),["credentials"],{userId:"alice"},
    {base:"https://backend.example",token:"secret"},async(_url,options)=>{
      calls++;redirect=options?.redirect;
      return new Response(null,{status:302,headers:{location:"https://evil.example"}});
    });
  assert.equal(redirect,"manual");assert.equal(calls,1);assert.equal(response.status,503);assert.equal(response.headers.get("location"),null);
});
