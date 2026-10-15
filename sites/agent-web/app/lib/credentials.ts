export type Provider = "model"|"serper"|"serpapi";
export type CredentialStatus = {provider:Provider;configured:boolean;last_four:string|null;updated_at:number|null;consent_at:number|null};
export type CredentialState = {provider_hostname:string;statuses:CredentialStatus[]};
export type CredentialApi = (path:string,body?:unknown)=>Promise<unknown>;

export async function submitCredentials(input:{providerHostname:string;modelKey:string;serperKey:string;serpapiKey:string;consent:boolean},api:CredentialApi):Promise<{handle:string|null}> {
  if (!input.providerHostname || !input.modelKey.trim()) throw new Error("credential_required");
  if (input.consent === true) {
    for (const [provider,key] of [["model",input.modelKey],["serper",input.serperKey],["serpapi",input.serpapiKey]])
      if (key) await api("credentials",{provider,key,consent:true});
    return {handle:null};
  }
  const response = await api("credential-sessions",{model_key:input.modelKey,
    ...(input.serperKey?{serper_key:input.serperKey}:{}),...(input.serpapiKey?{serpapi_key:input.serpapiKey}:{})}) as {handle?:string};
  if (!response || typeof response.handle !== "string" || !response.handle) throw new Error("invalid_credential_response");
  return {handle:response.handle};
}

export async function deleteCredential(provider:Provider,api:CredentialApi,onChange:(handle:null)=>void) {
  try { await api("credentials/delete",{provider}); }
  finally { onChange(null); }
}

export async function credentialApi(path:string,body?:unknown):Promise<unknown> {
  const response=await fetch(`/api/agent/${path}`,{method:body===undefined?"GET":"POST",cache:"no-store",
    ...(body===undefined?{}:{headers:{"content-type":"application/json"},body:JSON.stringify(body)})});
  const data=await response.json().catch(()=>({}));
  if(!response.ok) throw new Error("credential_request_failed");
  return data;
}
