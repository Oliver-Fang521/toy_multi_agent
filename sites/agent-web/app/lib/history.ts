export function completedMessage<E>(run:{status:string;result?:{answer:string;evidence:E[]}}) {
  return run.status === "completed" && run.result ?
    {role:"assistant" as const,content:run.result.answer,evidence:run.result.evidence} : null;
}

export async function loadConversationMessages<M>(fetchPage:(offset:number)=>Promise<{messages:M[];next_offset:number|null}>) {
  const messages:M[] = [];
  let offset:number|null = 0;
  while (offset !== null) {
    const page = await fetchPage(offset);
    messages.push(...page.messages);
    if (page.next_offset !== null && page.next_offset <= offset) throw new Error("invalid_history_page");
    offset = page.next_offset;
  }
  return messages;
}
