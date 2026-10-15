import test from "node:test";
import assert from "node:assert/strict";
import { submitCredentials, deleteCredential } from "../app/lib/credentials.ts";

const input = {providerHostname:"models.example",modelKey:"synthetic-browser-private-8314",serperKey:"",serpapiKey:"",consent:false};
test("unchecked consent uses only a temporary body, never URL or storage",async () => {
  const calls:{path:string;body:unknown}[]=[];
  const result=await submitCredentials(input,async(path,body)=>{calls.push({path,body});return {handle:"temporary-handle",expires_in:1800};});
  assert.deepEqual(result,{handle:"temporary-handle"});
  assert.deepEqual(calls,[{path:"credential-sessions",body:{model_key:input.modelKey}}]);
  assert.ok(!calls[0].path.includes(input.modelKey));
});
test("only explicit consent saves each provided provider",async () => {
  const calls:unknown[]=[];
  assert.deepEqual(await submitCredentials({...input,consent:true,serperKey:"synthetic-search-2241"},async(path,body)=>{calls.push({path,body});return {};}),{handle:null});
  assert.deepEqual(calls,[{path:"credentials",body:{provider:"model",key:input.modelKey,consent:true}},
    {path:"credentials",body:{provider:"serper",key:"synthetic-search-2241",consent:true}}]);
});
test("hostname is mandatory and failed save never returns configured state",async () => {
  let calls=0;
  await assert.rejects(submitCredentials({...input,providerHostname:""},async()=>{calls++;return {};}));
  assert.equal(calls,0);
  await assert.rejects(submitCredentials({...input,consent:true},async()=>{throw new Error("failed");}));
});
test("deletion clears temporary handle after success and on uncertain failure",async () => {
  let handle:string|null="temporary-handle";
  await deleteCredential("model",async(path,body)=>{assert.equal(path,"credentials/delete");assert.deepEqual(body,{provider:"model"});return {};},value=>{handle=value;});
  assert.equal(handle,null);
  handle="temporary-handle";
  await assert.rejects(deleteCredential("model",async()=>{throw new Error("offline");},value=>{handle=value;}));
  assert.equal(handle,null);
});
