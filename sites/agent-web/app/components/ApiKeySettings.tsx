"use client";
import { useEffect, useRef, useState } from "react";
import { KeyRound, X } from "lucide-react";
import { credentialApi, deleteCredential, submitCredentials } from "../lib/credentials";
import type { CredentialState, Provider } from "../lib/credentials";

export default function ApiKeySettings({onCredentialChange}:{onCredentialChange:(handle:string|null)=>void}) {
  const [open,setOpen]=useState(false);
  const [state,setState]=useState<CredentialState|null>(null);
  const [modelKey,setModelKey]=useState("");
  const [serperKey,setSerperKey]=useState("");
  const [serpapiKey,setSerpapiKey]=useState("");
  const [consent,setConsent]=useState(false);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  const [notice,setNotice]=useState("");
  const dialog=useRef<HTMLDialogElement>(null);
  const generation=useRef(0);
  function clear(){setModelKey("");setSerperKey("");setSerpapiKey("");setConsent(false);}
  function close(){generation.current++;clear();setOpen(false);}
  async function refresh(){const data=await credentialApi("credentials") as CredentialState;
    if(!data?.provider_hostname || !Array.isArray(data.statuses)) throw new Error("configuration_missing");
    setState(data);}
  useEffect(()=>{
    if(!open) return;
    dialog.current?.showModal();
    void refresh().catch(()=>setError("密钥服务暂未配置或不可用，请稍后重试。"));
    return ()=>dialog.current?.close();
  },[open]);
  async function save(event:React.FormEvent){
    event.preventDefault(); if(!state?.provider_hostname||busy)return;
    const current=generation.current;
    setBusy(true);setError("");setNotice("");
    try{
      const result=await submitCredentials({providerHostname:state.provider_hostname,modelKey,serperKey,serpapiKey,consent},credentialApi);
      if(current!==generation.current)return;
      clear();onCredentialChange(result.handle);
      setNotice(result.handle?"已启用临时密钥，30 分钟后失效；刷新后需重新输入。":"已加密保存。接收 Key 不代表模型访问能力已验证。");
      await refresh();
    }catch{if(current===generation.current){setError("保存失败或状态不确定，请查看当前配置后重试。不会使用站长的密钥。");await refresh().catch(()=>{});}}
    finally{setBusy(false);}
  }
  async function remove(provider:Provider){
    setBusy(true);setError("");setNotice("");clear();
    try{await deleteCredential(provider,credentialApi,onCredentialChange);await refresh();setNotice("已删除在线密钥并取消你的活动任务。已发送的请求可能仍计费；基础设施快照不会保证立即清除。");}
    catch{setError("删除未能确认，请刷新状态重试。需要立即撤销访问时，请在 API 服务商处撤销 Key。");}
    finally{setBusy(false);}
  }
  return <><button className="key-settings-trigger" onClick={()=>{clear();setState(null);setError("");setNotice("");setOpen(true);}}><KeyRound size={14}/> API keys</button>
    {open&&<dialog ref={dialog} className="key-settings" aria-labelledby="key-settings-title" onCancel={event=>{event.preventDefault();close();}}>
      <div className="key-settings-heading"><div><div className="eyebrow">YOUR ACCOUNT · YOUR KEYS</div><h2 id="key-settings-title">API 密钥设置</h2></div><button onClick={close} aria-label="关闭密钥设置"><X size={18}/></button></div>
      <p className="key-host">模型 Key 接收服务：<strong>{state?.provider_hostname||"正在读取服务端配置…"}</strong></p>
      <p className="key-disclosure">模型与工具费用计入你的 API 账号。Key 会发送至本服务后端及所显示的模型服务；勾选保存后将加密存储。未勾选时仅临时使用。</p>
      <p className="key-small">会员订阅与 API 余额是不同概念；可用模型、额度与计费取决于你的服务商。本站仍承担网页托管与后端费用。资源限额不是账单金额上限。</p>
      <div className="key-saved" aria-label="已保存的密钥">{state?.statuses.map(item=><div key={item.provider}><span>{item.provider}<small>{item.configured?`•••• ${item.last_four} · ${item.updated_at?new Date(item.updated_at*1000).toLocaleDateString():""}`:"未配置"}</small></span><button disabled={busy} onClick={()=>void remove(item.provider)} aria-label={`删除 ${item.provider} 密钥`}>删除</button></div>)}</div>
      <form onSubmit={save} autoComplete="off">
        <label>模型 API Key<input type="password" name="model-api-key" value={modelKey} onChange={e=>setModelKey(e.target.value)} maxLength={4096} required autoComplete="off" disabled={busy||!state?.provider_hostname}/></label>
        <details><summary>可选独立搜索服务</summary><p className="key-small">留空则本次临时执行不启用独立搜索；已保存搜索 Key 只在保存模式中使用。模型服务自带搜索仍计入模型账号。</p>
          <label>Serper Key（发送至 google.serper.dev）<input type="password" value={serperKey} onChange={e=>setSerperKey(e.target.value)} autoComplete="off" maxLength={4096} disabled={busy}/></label>
          <label>SerpAPI Key（发送至 serpapi.com）<input type="password" value={serpapiKey} onChange={e=>setSerpapiKey(e.target.value)} autoComplete="off" maxLength={4096} disabled={busy}/></label></details>
        <label className="key-consent"><input type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)} disabled={busy}/>保存本次填写的 Key，下次登录使用（服务端加密存储）</label>
        <button type="submit" className="key-save" disabled={busy||!state?.provider_hostname||!modelKey.trim()}>{busy?"处理中…":consent?"同意并加密保存":"仅临时使用 · 30 分钟"}</button>
      </form>
      {error&&<p role="alert" className="key-error">{error}</p>}{notice&&<p role="status" className="key-notice">{notice}</p>}
      <p className="key-small">服务器会接触并解密 Key；不保存于浏览器。关闭窗口清空输入。删除不保证物理擦除内存或备份。gpt-6.1-sol / low 是模型偏好，不保证你的账号可用。</p>
    </dialog>}</>;
}
