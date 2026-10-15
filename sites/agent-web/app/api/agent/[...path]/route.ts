import { getChatGPTUser } from "../../../chatgpt-auth";
import { forwardAgentRequest } from "../../../lib/proxy";

export const dynamic = "force-dynamic";
async function handle(request: Request, context: { params: Promise<{ path: string[] }> }) {
  return forwardAgentRequest(request, (await context.params).path, await getChatGPTUser(),
    { base: process.env.AGENT_BACKEND_URL, token: process.env.AGENT_SERVICE_TOKEN });
}
export const GET = handle;
export const POST = handle;
