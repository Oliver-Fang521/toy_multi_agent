"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUp, ArrowUpRight, BookOpen, Check, ChevronRight, Circle, FileText,
  GitBranch, LoaderCircle, Menu, Paperclip, Plus, Search, ShieldCheck, Square, X } from "lucide-react";
import { safeSourceUrl } from "../lib/proxy";
import { completedMessage, loadConversationMessages } from "../lib/history";
import ApiKeySettings from "./ApiKeySettings";
import { credentialApi } from "../lib/credentials";
import type { CredentialState } from "../lib/credentials";

type Evidence = { evidence_id: string; kind: string; summary: string; source_url?: string };
type Message = { role: "user" | "assistant"; content: string; evidence?: Evidence[] };
type Conversation = { id: string; created: number };
type Run = { id: string; conversation: string; status: string; question: string; error?: string;
  progress: { event: string; time: number }[]; result?: { answer: string; evidence: Evidence[] } };
const terminal = new Set(["completed", "failed", "cancelled", "interrupted"]);
const stages = ["Plan", "Research", "Verify", "Answer"];
const prompts = [
  { icon: Search, label: "Explore a topic", text: "What are the main differences between sodium-ion and lithium-ion batteries?" },
  { icon: FileText, label: "Work with a file", text: "Help me analyze the file I attach and identify the most important findings." },
  { icon: GitBranch, label: "Solve a problem", text: "A solar panel produces 420 W for 5 hours a day. How many kWh does it produce in 30 days?" },
];
const errors: Record<string,string> = {
  hourly_limit:"You’ve reached the limit of 3 requests per hour. Please try again later.",
  active_run_limit:"You already have an active request. Finish or cancel it first.",
  service_busy:"The research queue is full. Please try again shortly.",
  service_unreachable:"The research service is temporarily unreachable. Please try again.",
  service_not_configured:"The research service is not configured yet.",
  unsupported_file:"This file type isn’t supported. Try a PDF, text, CSV, XLSX, image or audio file.",
  file_too_large:"Files must be smaller than 32 MB.",
  timeout:"This research reached its time limit. Try a more focused question.",
  critic_rejection:"The agent couldn’t verify an answer with enough evidence. Try refining your question.",
  sign_in_required:"Please sign in to start a research request.",
};

export default function AgentChat({ signedIn, displayName, signInPath, signOutPath }:
  { signedIn:boolean; displayName:string; signInPath:string; signOutPath:string }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [cid, setCid] = useState<string|null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [file, setFile] = useState<File|null>(null);
  const [run, setRun] = useState<Run|null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [credentialHandle,setCredentialHandle] = useState<string|null>(null);
  const [savedCredential,setSavedCredential] = useState(false);
  const pollGeneration = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const attempt = useRef<{key:string;question:string;file:File|null;attachmentId:string|null}|null>(null);
  const activeKey = `toy-agent-active:${displayName}`;
  const busy = pending || Boolean(run && !terminal.has(run.status));
  const eventNames = new Set(run?.progress?.map(item => item.event));
  const stage = eventNames.has("submit_started") ? 3 : eventNames.has("verify_started") ? 2 : eventNames.has("act_started") ? 1 : 0;

  async function api<T>(path:string, options:RequestInit = {}): Promise<T> {
    const response = await fetch(`/api/agent/${path}`, { ...options, cache:"no-store" });
    const body = await response.json().catch(() => ({}));
    const failure = body as {error?:string;detail?:string};
    if (!response.ok) throw new Error(errors[failure.error || failure.detail || ""] || `Request failed (${response.status}). Please try again.`);
    return body as T;
  }
  async function refreshConversations() {
    try { setConversations(await api<Conversation[]>("conversations")); } catch(e) { setError((e as Error).message); }
  }
  async function refreshCredentials(){try{const state=await credentialApi("credentials") as CredentialState;
    setSavedCredential(Boolean(state.statuses?.some(s=>s.provider==="model"&&s.configured)));}catch{setSavedCredential(false);}}
  function credentialsChanged(handle:string|null){setCredentialHandle(handle);void refreshCredentials();}
  useEffect(()=>{if(signedIn)void refreshCredentials();else{setCredentialHandle(null);setSavedCredential(false);}},[signedIn]);
  useEffect(()=>{if(!credentialHandle)return;const timer=setTimeout(()=>setCredentialHandle(null),1800*1000);return()=>clearTimeout(timer);},[credentialHandle]);
  async function history(id:string) {
    return loadConversationMessages<Message>(offset => api<{messages:Message[];next_offset:number|null}>(`conversations/${id}?offset=${offset}`));
  }
  useEffect(() => {
    if (!signedIn) return;
    void refreshConversations();
    const identifier = localStorage.getItem(activeKey);
    if (identifier) void (async () => {
      try {
        const restored = await api<Run>(`runs/${identifier}`);
        const savedMessages = await history(restored.conversation);
        setCid(restored.conversation);
        setMessages(terminal.has(restored.status) ? savedMessages :
          [...savedMessages, {role:"user",content:restored.question}]);
        setRun(restored);
        if (terminal.has(restored.status)) localStorage.removeItem(activeKey);
      } catch { localStorage.removeItem(activeKey); }
    })();
  }, [signedIn]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior:"smooth" }); }, [messages.length, run?.status]);
  useEffect(() => {
    if (!run || terminal.has(run.status)) return;
    const generation = ++pollGeneration.current;
    let timer: ReturnType<typeof setTimeout>;
    let stopped = false;
    async function poll() {
      try {
        const next:Run = await api<Run>(`runs/${run!.id}`);
        if (stopped || generation !== pollGeneration.current) return;
        setRun(next);
        if (next.status === "completed" && next.result) {
          localStorage.removeItem(activeKey);
          setMessages(current => [...current, completedMessage(next)!]);
          void refreshConversations();
        } else if (terminal.has(next.status) && next.status !== "cancelled") {
          localStorage.removeItem(activeKey);
          setError(errors[next.error || ""] || "This request could not be completed. Try a more focused question.");
        } else if (!terminal.has(next.status)) timer = setTimeout(poll, 2000);
      } catch(e) {
        if (!stopped) { setError((e as Error).message); timer = setTimeout(poll, 5000); }
      }
    }
    timer = setTimeout(poll, 1000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [run?.id]);

  async function openConversation(id:string) {
    if (busy) return;
    try { const data = await history(id); setCid(id); setMessages(data); setRun(null); setError(""); setMobileMenu(false); }
    catch(e) { setError((e as Error).message); }
  }
  function newConversation() {
    if (busy) return;
    pollGeneration.current++; attempt.current = null; setCid(null); setMessages([]); setRun(null); setError(""); setQuestion(""); setFile(null); setMobileMenu(false);
  }
  async function submit() {
    if (!signedIn || busy || !question.trim()) return;
    if(!credentialHandle&&!savedCredential&&!attempt.current){setError("请先在 API keys 中配置你自己的密钥；历史答案仍可查看。");return;}
    setPending(true); setError("");
    const text = question.trim();
    if (!attempt.current || attempt.current.question !== text || attempt.current.file !== file)
      attempt.current = {key:crypto.randomUUID(), question:text, file, attachmentId:null};
    try {
      const identifier = cid || (await api<{id:string}>("conversations",{method:"POST"})).id;
      setCid(identifier);
      let attachmentId:null|string = attempt.current.attachmentId;
      if (file && !attachmentId) {
        if (file.size > 32 * 1024 * 1024) throw new Error(errors.file_too_large);
        const form = new FormData(); form.append("file", file);
        attachmentId = (await api<{id:string}>(`conversations/${identifier}/uploads`,{method:"POST",body:form})).id;
        attempt.current.attachmentId = attachmentId;
      }
      const next:Run = await api<Run>(`conversations/${identifier}/runs`, {method:"POST",
        headers:{"content-type":"application/json","idempotency-key":attempt.current.key,...(credentialHandle?{"x-agent-credential-handle":credentialHandle}:{})},
        body:JSON.stringify({question:text, attachment_id:attachmentId})});
      setMessages(current => [...current,{role:"user",content:text}]);
      if (next.status === "completed" && next.result) {
        setMessages(current => [...current,completedMessage(next)!]);
        localStorage.removeItem(activeKey);
      } else if (terminal.has(next.status)) {
        localStorage.removeItem(activeKey);
        if (next.status !== "cancelled") setError(errors[next.error || ""] || "This request could not be completed. Try a more focused question.");
      } else localStorage.setItem(activeKey, next.id);
      attempt.current = null; setRun(next); setQuestion(""); setFile(null); void refreshConversations();
    } catch(e) { setError((e as Error).message); }
    finally { setPending(false); }
  }
  async function cancel() {
    if (!run) return;
    try { const cancelled = await api<Run>(`runs/${run.id}/cancel`, {method:"POST"}); pollGeneration.current++; localStorage.removeItem(activeKey); setRun(cancelled); }
    catch(e) { setError((e as Error).message); }
  }

  return <div className="workspace">
    <aside className={`sidebar ${mobileMenu ? "open" : ""}`}>
      <a className="brand" href="/"><span className="brand-mark"><GitBranch size={21}/></span><span>toy<span className="brand-light"> / agent</span></span></a>
      <button className="new-chat" onClick={newConversation} disabled={busy}><Plus size={17}/> New conversation <span>↗</span></button>
      <div className="sidebar-caption">WORKSPACE</div>
      <button className="nav-current" onClick={newConversation}><BookOpen size={17}/> Research assistant</button>
      <div className="sidebar-caption history-caption">YOUR CONVERSATIONS</div>
      <nav className="history-list" aria-label="Conversation history">
        {conversations.length ? conversations.map((item,index) => <button key={item.id} className={cid===item.id?"selected":""}
          onClick={() => openConversation(item.id)} disabled={busy}><Circle size={7}/><span>Conversation {conversations.length-index}</span><ChevronRight size={13}/></button>)
          : <p className="history-empty">Your research starts here.<br/>Conversations will appear as you go.</p>}
      </nav>
      <div className="sidebar-bottom"><div className="privacy-note"><ShieldCheck size={16}/><span>Your questions and files<br/>stay in your workspace.</span></div>
        {signedIn ? <div className="profile"><span className="avatar">{displayName[0]?.toUpperCase()}</span><span className="profile-name">{displayName}<small>Personal workspace</small></span><a href={signOutPath} target="_top" aria-label="Sign out">↗</a></div>
        : <a className="sidebar-signin" href={signInPath} target="_top">Sign in with ChatGPT <ArrowUpRight size={15}/></a>}</div>
    </aside>
    <main className="main-panel">
      <header className="topbar"><button className="mobile-toggle" onClick={() => setMobileMenu(!mobileMenu)} aria-label="Toggle navigation"><Menu size={20}/></button>
        <span>Research assistant <span className="topbar-separator">/</span> <span className="muted">{cid?"Conversation":"New conversation"}</span></span>
        {signedIn&&<ApiKeySettings onCredentialChange={credentialsChanged}/>}
        <div className="model-pill"><span className="status-dot"/> GPT-6.1 Sol <span>· low</span></div></header>
      <section className={`conversation ${messages.length ? "has-messages" : ""}`}>
        {!messages.length && <div className="welcome"><div className="eyebrow"><span/> A LITTLE TEAM FOR BIG QUESTIONS</div>
          <h1>Curiosity, meet<br/><span>a second pair of eyes.</span></h1>
          <p className="intro">Ask a question, share a file, or untangle a problem.<br/>Your agent researches, checks, and brings the sources.</p>
          <div className="prompt-grid">{prompts.map(item => <button key={item.label} onClick={() => setQuestion(item.text)} disabled={busy}>
            <item.icon size={20}/><strong>{item.label}</strong><span>{item.text}</span><ArrowUpRight size={16} className="prompt-arrow"/></button>)}</div>
          <div className="method-strip"><span><Search size={14}/> Research</span><span>→</span><span><ShieldCheck size={14}/> Verify</span><span>→</span><span><Check size={14}/> Answer with sources</span></div></div>}
        <div className="messages">{messages.map((message,index) => <article key={index} className={`message ${message.role}`}>
          <div className="message-label">{message.role==="user"?"YOU":<><GitBranch size={15}/> TOY AGENT</>}</div><p className="message-content">{message.content}</p>
          {message.evidence?.length ? <div className="evidence-section"><div className="evidence-heading"><BookOpen size={14}/> Supporting evidence</div>
            {message.evidence.map(evidence => <div className="source-card" key={evidence.evidence_id}><span className="source-number">{evidence.evidence_id.split("-").at(-1)}</span>
              <div><p>{evidence.summary}</p>{evidence.source_url && safeSourceUrl(evidence.source_url) ? <a href={safeSourceUrl(evidence.source_url)!} target="_blank" rel="noopener noreferrer">{new URL(evidence.source_url).hostname}<ArrowUpRight size={12}/></a> : <small>{evidence.kind} evidence</small>}</div></div>)}</div>:null}
        </article>)}</div>
        {run && <div className="run-status" aria-live="polite"><div className="run-heading">{terminal.has(run.status)?<Check size={15}/>:<LoaderCircle size={15} className="spin"/>}
          {run.status==="completed"?"Research complete":run.status==="cancelled"?"Request cancelled":run.status==="queued"?"Waiting for a research slot":run.status==="failed"||run.status==="interrupted"?"Request ended":"Your agent is working"}
          {!terminal.has(run.status)&&<button onClick={cancel}><Square size={10}/> Stop</button>}</div>
          <div className="stage-track">{stages.map((label,index)=><span key={label} className={index<=stage?"reached":""}><i>{index<stage||run.status==="completed"?<Check size={10}/>:index+1}</i>{label}</span>)}</div></div>}
        <div ref={bottom}/>
      </section>
      <footer className="composer-area">
        {error&&<div className="error-banner" role="alert">{error}<button onClick={()=>setError("")} aria-label="Dismiss error"><X size={14}/></button></div>}
        <form className="composer" onSubmit={event=>{event.preventDefault();void submit();}}>
          {file&&<div className="attachment-chip"><FileText size={13}/>{file.name}<button type="button" onClick={()=>setFile(null)} aria-label="Remove attachment"><X size={13}/></button></div>}
          <textarea aria-label="Your question" placeholder="What would you like to figure out?" value={question} maxLength={12000} rows={2}
            onChange={event=>setQuestion(event.target.value)} onKeyDown={event=>{if(event.key==="Enter"&&!event.shiftKey&&!event.nativeEvent.isComposing){event.preventDefault();void submit();}}} disabled={busy}/>
          <div className="composer-actions"><label className="attach-button" title="Attach one file, up to 32 MB"><Paperclip size={18}/><input type="file" disabled={busy||!signedIn}
            accept=".pdf,.txt,.csv,.xlsx,.png,.jpg,.jpeg,.webp,.mp3,.wav,.m4a" onChange={event=>setFile(event.target.files?.[0]||null)}/></label>
            <span>One question. A whole team behind it.</span>
            {signedIn?<button className="send-button" type="submit" disabled={busy||!question.trim()} aria-label="Send question">{pending?<LoaderCircle size={18} className="spin"/>:<ArrowUp size={20}/>}</button>
              :<a className="signin-button" href={signInPath} target="_top">Sign in to start <ArrowUpRight size={15}/></a>}</div>
        </form><p className="composer-footnote">3 requests per hour · Sources you can inspect · Answers may need your judgment</p>
      </footer>
    </main>
  </div>;
}
