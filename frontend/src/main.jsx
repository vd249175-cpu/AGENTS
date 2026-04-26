import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Bot,
  CheckCircle2,
  Database,
  MessageSquare,
  Network,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Send,
  Settings,
  Square,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import "./styles.css";

const DEFAULT_STREAM_MODE = ["messages", "updates", "custom"];
const DEFAULT_SPACE_COLOR = "#2563eb";

function pretty(value) {
  return JSON.stringify(value ?? null, null, 2);
}

function parseJson(text, label) {
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} 不是合法 JSON: ${error.message}`);
  }
}

function shortId(prefix = "") {
  return `${prefix}${Math.random().toString(36).slice(2, 10)}`;
}

function makeChatMessage(role, content, extra = {}) {
  return {
    id: shortId("msg-"),
    created_at: new Date().toISOString(),
    role,
    content,
    ...extra,
  };
}

function normalizeChatMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages.map((message, index) => ({
    id: message.id || `legacy-${index}-${String(message.role || "message").replace(/\s+/g, "-")}`,
    created_at: message.created_at || message.createdAt || null,
    ...message,
  }));
}

function extractLastContent(value) {
  let last = "";
  function visit(node) {
    if (!node) return;
    if (Array.isArray(node)) {
      node.forEach(visit);
      return;
    }
    if (typeof node === "object") {
      if (typeof node.content === "string" && node.content.trim()) last = node.content;
      Object.values(node).forEach(visit);
    }
  }
  visit(value);
  return last || pretty(value);
}

function contentPreview(value, limit = 260) {
  if (value == null) return "";
  if (typeof value === "string") return value.slice(0, limit);
  if (Array.isArray(value)) {
    return value.map((item) => contentPreview(item, limit)).filter(Boolean).join("").slice(0, limit);
  }
  if (typeof value === "object") {
    if (typeof value.text === "string") return value.text.slice(0, limit);
    if (typeof value.content === "string") return value.content.slice(0, limit);
    return pretty(value).slice(0, limit);
  }
  return String(value).slice(0, limit);
}

function normalizeStreamChunk(chunk) {
  if (chunk && typeof chunk === "object" && !Array.isArray(chunk)) {
    const type = typeof chunk.type === "string" ? chunk.type : null;
    if (type) return { type, data: chunk.data };
  }
  if (Array.isArray(chunk) && chunk.length === 2) {
    const [first, second] = chunk;
    if (typeof first === "string") return { type: first, data: second };
    if (typeof second === "string") return { type: second, data: first };
  }
  return { type: "raw", data: chunk };
}

function findToolCalls(value) {
  const calls = [];
  function visit(node) {
    if (!node) return;
    if (Array.isArray(node)) {
      node.forEach(visit);
      return;
    }
    if (typeof node === "object") {
      const rawCalls = node.tool_calls || node.toolCalls;
      if (Array.isArray(rawCalls)) {
        rawCalls.forEach((call) => calls.push(call?.name || call?.function?.name || call?.tool || "tool_call"));
      }
      Object.values(node).forEach(visit);
    }
  }
  visit(value);
  return [...new Set(calls)].filter(Boolean);
}

function summarizeStreamChunk(chunk, index) {
  const normalized = normalizeStreamChunk(chunk);
  const { type, data } = normalized;
  if (type === "custom") {
    if (!data?.tool && data?.type !== "tool") return null;
    const toolName = data?.tool || data?.name || "tool";
    return {
      id: `${index}-custom`,
      type: "tool",
      title: `工具 · ${toolName}`,
      detail: data?.preview || data?.result || data?.error || "",
    };
  }
  if (type === "updates") return null;
  if (type === "messages") {
    const calls = findToolCalls(data);
    const toolMessages = collectMessageNodes(data).filter((message) => messageRole(message) === "tool");
    if (!calls.length && !toolMessages.length) return null;
    if (toolMessages.length) {
      return {
        id: `${index}-tool-message`,
        type: "tool",
        title: "工具返回",
        detail: toolMessages.map((message) => contentPreview(message.content)).filter(Boolean).join("\n\n"),
      };
    }
    return {
      id: `${index}-messages`,
      type: "tool",
      title: `工具调用 · ${calls.join(", ")}`,
      detail: "",
    };
  }
  return null;
}

function buildTraceEvents(result) {
  const chunks = result?.response?.chunks || result?.chunks || [];
  return chunks.map((chunk, index) => summarizeStreamChunk(chunk, index)).filter(Boolean);
}

function isVisibleTraceEvent(eventItem) {
  const type = String(eventItem?.type || "").toLowerCase();
  const title = String(eventItem?.title || "").toLowerCase();
  return type === "tool" || type === "mail" || title.includes("tool") || title.includes("工具");
}

function messageRole(node) {
  const role = String(node?.role || node?.type || node?.message_type || "").toLowerCase();
  if (role === "ai" || role === "assistant" || role === "aimessage" || role === "aimessagechunk") return "assistant";
  if (role === "human" || role === "user" || role === "humanmessage") return "user";
  if (role === "tool" || role === "toolmessage") return "tool";
  return role || "message";
}

function collectMessageNodes(value) {
  const nodes = [];
  function visit(node) {
    if (!node) return;
    if (Array.isArray(node)) {
      node.forEach(visit);
      return;
    }
    if (typeof node === "object") {
      const role = messageRole(node);
      if ((node.content != null || node.tool_calls || node.toolCalls) && ["assistant", "user", "tool"].includes(role)) {
        nodes.push(node);
        return;
      }
      Object.values(node).forEach(visit);
    }
  }
  visit(value);
  return nodes;
}

function eventTime(value) {
  const raw = value?.at || value?.finished_at || value?.started_at || "";
  const time = raw ? Date.parse(raw) : Number.NaN;
  return Number.isFinite(time) ? time : 0;
}

function eventClock(value) {
  const raw = value?.at || value?.finished_at || value?.started_at || "";
  if (!raw) return "";
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return "";
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function summarizeAgentEvent(eventItem, agentName, index) {
  const eventName = eventItem?.event || eventItem?.type || "event";
  const at = eventItem?.at || eventItem?.finished_at || eventItem?.started_at || "";
  const clock = eventClock(eventItem);
  const messages = collectMessageNodes(eventItem?.state?.messages || eventItem?.state || []);
  const toolCalls = messages.flatMap((message) => findToolCalls(message).map((name) => ({ name, message })));
  const toolMessages = messages.filter((message) => messageRole(message) === "tool");
  const assistantMessages = messages
    .filter((message) => messageRole(message) === "assistant")
    .filter((message) => contentPreview(message.content).trim());

  if (toolCalls.length) {
    const names = [...new Set(toolCalls.map((call) => call.name))].join(", ");
    return {
      id: `event-${agentName}-${at || index}-tool-call-${names}`,
      kind: "tool",
      role: "behavior",
      at,
      time: eventTime(eventItem),
      content: `${agentName} 调用工具`,
      events: [{ id: `tool-call-${index}`, type: "tool", title: names, detail: "" }],
    };
  }

  if (toolMessages.length) {
    const detail = toolMessages.map((message) => contentPreview(message.content)).filter(Boolean).join("\n\n");
    return {
      id: `event-${agentName}-${at || index}-tool-result`,
      kind: "tool",
      role: "behavior",
      at,
      time: eventTime(eventItem),
      content: `${agentName} 工具结果`,
      events: [{ id: `tool-result-${index}`, type: "tool", title: "工具返回", detail }],
    };
  }

  if (eventName === "mail_wake_success") {
    return {
      id: `event-${agentName}-${at || index}-wake-message`,
      kind: "message",
      role: "agent",
      at,
      time: eventTime(eventItem),
      content: eventItem.reply || "",
      events: [],
    };
  }

  if (assistantMessages.length && eventName === "end" && eventItem.phase === "after_agent") {
    const detail = contentPreview(assistantMessages[assistantMessages.length - 1].content, 600);
    return {
      id: `event-${agentName}-${at || index}-assistant-message`,
      kind: "message",
      role: "agent",
      at,
      time: eventTime(eventItem),
      content: detail,
      events: [],
    };
  }

  return null;
}

function summarizeMailEvent(item, selectedAgent, index) {
  const message = item?.message || {};
  const isReceived = message.to === selectedAgent;
  const direction = isReceived ? "收到邮件" : "发送邮件";
  const peer = message.to === selectedAgent ? message.from : message.to;
  return {
    id: `mail-${item?.at || index}-${message.message_id || index}`,
    kind: "tool",
    role: "behavior",
    at: item?.at || "",
    time: eventTime(item),
    content: `${selectedAgent} ${direction}`,
    events: [
      {
        id: `mail-detail-${index}`,
        type: "mail",
        title: peer ? `${isReceived ? "from" : "to"} ${peer}` : `${eventClock(item)} ${message.type || "message"}`.trim(),
        detail:
          typeof message.content === "string"
            ? message.content
            : contentPreview(message.content || message),
      },
    ],
  };
}

function normalizeSpaces(communication) {
  return Array.isArray(communication?.spaces) ? communication.spaces : [];
}

function agentPosition(agent, index, positions) {
  const saved = positions?.[agent.agent_name];
  if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) return saved;
  return { x: 120 + (index % 3) * 260, y: 120 + Math.floor(index / 3) * 170 };
}

function edgeKey(left, right) {
  return [left, right].sort().join("::");
}

function buildEdges(spaces, agentNames) {
  const allowed = new Set(agentNames);
  const edgeMap = new Map();
  for (const space of spaces) {
    const members = (space.members || []).filter((name) => allowed.has(name));
    for (let i = 0; i < members.length; i += 1) {
      for (let j = i + 1; j < members.length; j += 1) {
        const key = edgeKey(members[i], members[j]);
        const existing = edgeMap.get(key) || { from: members[i], to: members[j], spaces: [] };
        existing.spaces.push(space.id);
        edgeMap.set(key, existing);
      }
    }
  }
  return [...edgeMap.values()];
}

function useApi(baseUrl) {
  return useMemo(() => {
    const root = (baseUrl || "/api").replace(/\/$/, "");
    return async function request(path, options = {}) {
      const response = await fetch(`${root}${path}`, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
      });
      const text = await response.text();
      const data = text ? JSON.parse(text) : {};
      if (!response.ok) {
        const detail = data?.detail || text || response.statusText;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return data;
    };
  }, [baseUrl]);
}

function App() {
  const [baseUrl, setBaseUrl] = useState(localStorage.getItem("mainServerUrl") || "/api");
  const api = useApi(baseUrl);

  const [agents, setAgents] = useState([]);
  const [communication, setCommunication] = useState({ spaces: [] });
  const [agentPositions, setAgentPositions] = useState({});
  const [chatSessions, setChatSessions] = useState({});
  const [selectedAgent, setSelectedAgent] = useState("");
  const [selectedSpaceId, setSelectedSpaceId] = useState("");
  const [activeTab, setActiveTab] = useState("graph");
  const [status, setStatus] = useState("Ready");
  const [online, setOnline] = useState(false);
  const [busy, setBusy] = useState(false);

  const [newAgentName, setNewAgentName] = useState("");
  const [sourceAgent, setSourceAgent] = useState("SeedAgent");
  const [newSpaceName, setNewSpaceName] = useState("");

  const [chatMessages, setChatMessages] = useState([]);
  const [lastTraceEvents, setLastTraceEvents] = useState([]);
  const [chatText, setChatText] = useState("");
  const [attachmentUrl, setAttachmentUrl] = useState("");
  const [sendMode, setSendMode] = useState("direct");
  const [threadId, setThreadId] = useState("default");
  const [runId, setRunId] = useState("");
  const [streamMode, setStreamMode] = useState(pretty(DEFAULT_STREAM_MODE));

  const [agentRole, setAgentRole] = useState("");
  const [agentDescription, setAgentDescription] = useState("");
  const [agentResponsibilities, setAgentResponsibilities] = useState("");
  const [systemPromptExtra, setSystemPromptExtra] = useState("");
  const [defaultRunId, setDefaultRunId] = useState("");
  const [knowledgeRunId, setKnowledgeRunId] = useState("");
  const [brainPrompt, setBrainPrompt] = useState("");
  const [mainConfigJson, setMainConfigJson] = useState("{}");
  const [runtimeConfigJson, setRuntimeConfigJson] = useState("{}");
  const [monitor, setMonitor] = useState({ agents: [], mailbox_counts: {}, recent_mail: [] });

  const [dragging, setDragging] = useState(null);
  const [graphGesture, setGraphGesture] = useState(null);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [selectionBox, setSelectionBox] = useState(null);
  const latestPositionsRef = useRef({});
  const graphCanvasRef = useRef(null);
  const messagesRef = useRef(null);

  const spaces = useMemo(() => normalizeSpaces(communication), [communication]);
  const agentNames = agents.map((agent) => agent.agent_name);
  const edges = useMemo(() => buildEdges(spaces, agentNames), [spaces, agentNames]);
  const selectedSummary = agents.find((agent) => agent.agent_name === selectedAgent);
  const selectedRegistered = Boolean(selectedSummary?.registered);
  const selectedServiceReady =
    selectedRegistered &&
    Boolean(selectedSummary?.metadata?.service_url) &&
    !["starting", "stopped", "error"].includes(selectedSummary?.status);
  const sourceOptions = useMemo(() => {
    const names = agents.map((agent) => agent.agent_name);
    return [...new Set(["SeedAgent", "KnowledgeSeedAgent", ...names])];
  }, [agents]);
  const selectedSpace = spaces.find((space) => space.id === selectedSpaceId) || spaces[0] || null;
  const selectedAgentActivity = useMemo(() => {
    if (!selectedAgent) return [];
    const selectedMonitorAgent = (monitor.agents || []).find((agent) => agent.agent_name === selectedAgent);
    const eventItems = (selectedMonitorAgent?.events || [])
      .slice(-80)
      .map((eventItem, index) => summarizeAgentEvent(eventItem, selectedAgent, index))
      .filter(Boolean)
      .map((item, index) => ({ ...item, order: index }));
    const mailItems = (monitor.recent_mail || [])
      .filter((item) => item?.message?.from === selectedAgent || item?.message?.to === selectedAgent)
      .slice(-40)
      .map((item, index) => summarizeMailEvent(item, selectedAgent, index))
      .map((item, index) => ({ ...item, order: 1000 + index }));
    const seen = new Set();
    return [...eventItems, ...mailItems]
      .sort((left, right) => {
        if (left.time && right.time && left.time !== right.time) return left.time - right.time;
        return left.order - right.order;
      })
      .filter((item) => {
        const key =
          item.kind === "message"
            ? `message:${String(item.content || "").trim()}`
            : `tool:${item.content}:${item.events?.[0]?.title || ""}:${item.events?.[0]?.detail || ""}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .slice(-80);
  }, [monitor, selectedAgent]);

  async function run(label, operation, onError) {
    setBusy(true);
    setStatus(label);
    try {
      await operation();
      setStatus("Done");
    } catch (error) {
      setStatus(`Error: ${error.message}`);
      if (onError) onError(error);
    } finally {
      setBusy(false);
    }
  }

  async function refreshAgents() {
    await run("Refreshing agents", async () => {
      const result = await api("/admin/agents/available");
      const list = result.agents || [];
      const ui = result.ui || {};
      setAgents(list);
      setCommunication(result.communication || { spaces: [] });
      setAgentPositions(ui.agent_positions || {});
      latestPositionsRef.current = ui.agent_positions || {};
      setChatSessions(ui.chat_sessions || {});
      setOnline(true);
      const nextAgent = selectedAgent || list.find((agent) => agent.agent_name === "SeedAgent")?.agent_name || list[0]?.agent_name || "";
      if (nextAgent) {
        setSelectedAgent(nextAgent);
        await loadAgent(nextAgent, ui.chat_sessions || {});
      }
      const nextSpace = selectedSpaceId || result.communication?.spaces?.[0]?.id || "";
      setSelectedSpaceId(nextSpace);
    });
  }

  async function refreshMonitor() {
    try {
      const result = await api("/admin/monitor");
      setMonitor(result);
      const latestByName = new Map((result.agents || []).map((agent) => [agent.agent_name, agent]));
      setAgents((current) =>
        current.map((agent) => (latestByName.has(agent.agent_name) ? { ...agent, ...latestByName.get(agent.agent_name) } : agent)),
      );
    } catch {
      // Monitoring is opportunistic; the main health indicator covers connection loss.
    }
  }

  async function saveUi(nextPositions = agentPositions, nextChatSessions = chatSessions) {
    const result = await api("/admin/ui-state", {
      method: "PUT",
      body: JSON.stringify({
        agent_positions: nextPositions,
        chat_sessions: nextChatSessions,
      }),
    });
    setAgentPositions(result.ui.agent_positions || {});
    latestPositionsRef.current = result.ui.agent_positions || {};
    setChatSessions(result.ui.chat_sessions || {});
  }

  async function saveCommunication(nextCommunication) {
    const result = await api("/admin/communication", {
      method: "PUT",
      body: JSON.stringify(nextCommunication),
    });
    setCommunication(result.communication || nextCommunication);
    await refreshAgents();
  }

  async function loadAgent(agentName, sessions = chatSessions) {
    const [chat, main, runtimeResult, brainResult] = await Promise.allSettled([
      api(`/user/chat/config/${agentName}`),
      api(`/admin/agents/${agentName}/config`),
      api(`/admin/agents/${agentName}/runtime-config`),
      api(`/admin/agents/${agentName}/brain`),
    ]);

    if (chat.status === "rejected") throw chat.reason;
    if (main.status === "rejected") throw main.reason;

    const chatConfig = chat.value.chat || {};
    setThreadId(chatConfig.thread_id || "default");
    setRunId(chatConfig.run_id || "");
    setStreamMode(pretty(chatConfig.stream_mode || DEFAULT_STREAM_MODE));
    setMainConfigJson(pretty(main.value.config || {}));

    const runtimeConfig = runtimeResult.status === "fulfilled" ? runtimeResult.value.config || {} : {};
    setRuntimeConfigJson(pretty(runtimeConfig));
    setAgentRole(runtimeConfig.agentRole || "");
    setAgentDescription(runtimeConfig.agentDescription || "");
    setAgentResponsibilities((runtimeConfig.agentResponsibilities || []).join("\n"));
    setSystemPromptExtra(runtimeConfig.systemPromptExtra || "");
    setDefaultRunId(runtimeConfig.defaultRunId || "");
    setKnowledgeRunId(runtimeConfig.knowledgeRunId || "");
    setBrainPrompt(brainResult.status === "fulfilled" ? brainResult.value.content || "" : "");
    const session = sessions?.[agentName] || {};
    setChatMessages(normalizeChatMessages(session.messages));
    setLastTraceEvents(session.last_trace_events || session.lastTraceEvents || []);
  }

  useEffect(() => {
    localStorage.setItem("mainServerUrl", baseUrl);
    api("/healthz")
      .then(() => {
        setOnline(true);
        return refreshAgents();
      })
      .catch(() => setOnline(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl]);

  useEffect(() => {
    refreshMonitor();
    const timer = window.setInterval(refreshMonitor, 2500);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl]);

  useEffect(() => {
    if (activeTab !== "chat") return;
    const node = messagesRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [activeTab, chatMessages, selectedAgentActivity]);

  function selectAgent(agentName, tab = "chat") {
    setSelectedAgent(agentName);
    setActiveTab(tab);
    run(`Loading ${agentName}`, () => loadAgent(agentName));
  }

  function selectGraphAgent(agentName) {
    setSelectedAgent(agentName);
    loadAgent(agentName).catch((error) => setStatus(`Error: ${error.message}`));
  }

  function createAgent(event) {
    event.preventDefault();
    const name = newAgentName.trim();
    if (!name) return;
    run(`Creating ${name}`, async () => {
      await api("/admin/agents/create", {
        method: "POST",
        body: JSON.stringify({
          agent_name: name,
          source_agent: sourceAgent || "SeedAgent",
          overwrite: false,
        }),
      });
      const nextPositions = {
        ...agentPositions,
        [name]: { x: 160 + agents.length * 36, y: 140 + agents.length * 28 },
      };
      await saveUi(nextPositions, chatSessions);
      setNewAgentName("");
      setSelectedAgent(name);
      await refreshAgents();
    });
  }

  function deleteAgent() {
    if (!selectedAgent) return;
    if (!window.confirm(`删除 ${selectedAgent} 的本地 Agent 目录？这个操作不能撤销。`)) return;
    run(`Deleting ${selectedAgent}`, async () => {
      await api(`/admin/agents/${selectedAgent}`, { method: "DELETE" });
      setSelectedAgent("");
      setChatMessages([]);
      await refreshAgents();
    });
  }

  function startAgentService() {
    if (!selectedAgent) return;
    run(`Starting ${selectedAgent}`, async () => {
      await api(`/admin/agents/${selectedAgent}/service/start`, {
        method: "POST",
        body: JSON.stringify({
          host: "127.0.0.1",
          main_server_url: "http://127.0.0.1:8000",
        }),
      });
      for (let index = 0; index < 40; index += 1) {
        await new Promise((resolve) => setTimeout(resolve, 500));
        const result = await api("/admin/agents/available");
        const list = result.agents || [];
        setAgents(list);
        const current = list.find((agent) => agent.agent_name === selectedAgent);
        if (current?.registered && current?.metadata?.service_url) {
          await loadAgent(selectedAgent);
          return;
        }
      }
      throw new Error("AgentServer 启动请求已发送，但还没有完成注册；请稍后刷新。");
    });
  }

  function stopAgentService() {
    if (!selectedAgent) return;
    run(`Stopping ${selectedAgent}`, async () => {
      await api(`/admin/agents/${selectedAgent}/service/stop`, { method: "POST" });
      await refreshAgents();
    });
  }

  function clearRuntime() {
    if (!selectedAgent) return;
    run("Clearing checkpoints", async () => {
      await api(`/admin/agents/${selectedAgent}/runtime/clear`, {
        method: "POST",
        body: JSON.stringify({
          include_store: false,
          include_mail: false,
          include_knowledge: false,
          include_checkpoints: true,
        }),
      });
      const nextSessions = { ...chatSessions };
      delete nextSessions[selectedAgent];
      setChatSessions(nextSessions);
      setChatMessages([]);
      setLastTraceEvents([]);
      setMonitor((current) => ({
        ...current,
        agents: (current.agents || []).map((agent) =>
          agent.agent_name === selectedAgent ? { ...agent, events: [], last_event: null } : agent,
        ),
        recent_mail: (current.recent_mail || []).filter(
          (item) => item?.message?.from !== selectedAgent && item?.message?.to !== selectedAgent,
        ),
      }));
      await saveUi(agentPositions, nextSessions);
      await refreshMonitor();
    });
  }

  function persistChatMessages(agentName, nextMessages, traceEvents = lastTraceEvents) {
    const nextSessions = {
      ...chatSessions,
      [agentName]: {
        ...(chatSessions[agentName] || {}),
        messages: nextMessages,
        last_trace_events: traceEvents,
      },
    };
    setChatSessions(nextSessions);
    saveUi(agentPositions, nextSessions).catch(() => {});
  }

  function saveChatConfig() {
    if (!selectedAgent) return;
    run("Saving chat config", async () => {
      await api(`/user/chat/config/${selectedAgent}`, {
        method: "PUT",
        body: JSON.stringify({
          thread_id: threadId || "default",
          run_id: runId || null,
          stream_mode: parseJson(streamMode, "stream_mode"),
          version: "v2",
        }),
      });
    });
  }

  function newThread() {
    const next = `thread-${shortId()}`;
    setThreadId(next);
    setChatMessages([]);
    setLastTraceEvents([]);
    persistChatMessages(selectedAgent, [], []);
    run("Saving new thread", async () => {
      await api(`/user/chat/config/${selectedAgent}`, {
        method: "PUT",
        body: JSON.stringify({
          thread_id: next,
          run_id: null,
          stream_mode: parseJson(streamMode, "stream_mode"),
          version: "v2",
        }),
      });
    });
  }

  function sendChat(event) {
    event.preventDefault();
    if (!selectedAgent) return;
    const text = chatText.trim();
    const attachment = attachmentUrl.trim();
    if (!text && !attachment) return;
    const userMessage = makeChatMessage("user", text || attachment);
    const optimistic = [...chatMessages, userMessage];
    setChatMessages(optimistic);
    persistChatMessages(selectedAgent, optimistic);
    setChatText("");
    setAttachmentUrl("");

    run("Sending message", async () => {
      if (sendMode === "direct" && !selectedServiceReady) {
        throw new Error("当前 AgentServer 还没启动注册。请先点右上角“启动服务”，或改用“发邮件”。");
      }
      const body = {
        agent_name: selectedAgent,
        mode: sendMode,
        from: "user",
        text: text || null,
        thread_id: threadId,
        run_id: runId || null,
        stream_mode: parseJson(streamMode, "stream_mode"),
      };
      if (attachment) body.attachments = [{ link: attachment, summary: "user attachment" }];
      const result = await api("/user/chat", {
        method: "POST",
        body: JSON.stringify(body),
      });
      const traceEvents =
        result.mode === "mail"
          ? [
              {
                id: `mail-${result.message_id}`,
                type: "mail",
                title: result.wake_scheduled ? "mail delivered and wake scheduled" : "mail delivered",
                detail: `message_id=${result.message_id}`,
              },
            ]
          : buildTraceEvents(result);
      setLastTraceEvents(traceEvents);
      const reply =
        result.mode === "mail"
          ? result.wake_scheduled
            ? `Mail delivered and wake scheduled: ${result.message_id}`
            : `Mail delivered: ${result.message_id}`
          : result.response?.reply || extractLastContent(result.response || result);
      const nextMessages = [...optimistic, makeChatMessage("agent", reply, { events: traceEvents })];
      setChatMessages(nextMessages);
      const nextSessions = {
        ...chatSessions,
        [selectedAgent]: {
          ...(chatSessions[selectedAgent] || {}),
          messages: nextMessages,
          last_trace_events: traceEvents,
        },
      };
      setChatSessions(nextSessions);
      saveUi(agentPositions, nextSessions).catch(() => {});
      await refreshMonitor();
    }, (error) => {
      const nextMessages = [...optimistic, makeChatMessage("agent error", error.message)];
      setChatMessages(nextMessages);
      persistChatMessages(selectedAgent, nextMessages);
    });
  }

  function savePrompt() {
    if (!selectedAgent) return;
    run("Saving prompt", async () => {
      await api(`/admin/agents/${selectedAgent}/runtime-config`, {
        method: "PATCH",
        body: JSON.stringify({
          agentRole,
          agentDescription,
          agentResponsibilities: agentResponsibilities.split("\n").map((item) => item.trim()).filter(Boolean),
          systemPromptExtra,
          defaultRunId,
          knowledgeRunId: knowledgeRunId || null,
        }),
      });
      await loadAgent(selectedAgent);
    });
  }

  function saveBrainPrompt() {
    if (!selectedAgent) return;
    run("Saving brain prompt", async () => {
      const result = await api(`/admin/agents/${selectedAgent}/brain`, {
        method: "PUT",
        body: JSON.stringify({ content: brainPrompt }),
      });
      setBrainPrompt(result.content || "");
    });
  }

  function saveMainConfig() {
    if (!selectedAgent) return;
    run("Saving MainServer config", async () => {
      await api(`/admin/agents/${selectedAgent}/config`, {
        method: "PUT",
        body: JSON.stringify(parseJson(mainConfigJson, "MainServer config")),
      });
      await loadAgent(selectedAgent);
    });
  }

  function saveRuntimeConfig() {
    if (!selectedAgent) return;
    run("Saving runtime config", async () => {
      await api(`/admin/agents/${selectedAgent}/runtime-config`, {
        method: "PUT",
        body: JSON.stringify(parseJson(runtimeConfigJson, "runtime config")),
      });
      await loadAgent(selectedAgent);
    });
  }

  function addSpace() {
    const name = newSpaceName.trim() || `Space ${spaces.length + 1}`;
    const nextSpace = {
      id: shortId("space-"),
      name,
      color: DEFAULT_SPACE_COLOR,
      members: [],
    };
    setNewSpaceName("");
    setSelectedSpaceId(nextSpace.id);
    saveCommunication({ spaces: [...spaces, nextSpace] });
  }

  function deleteSelectedSpace() {
    if (!selectedSpace) return;
    saveCommunication({ spaces: spaces.filter((space) => space.id !== selectedSpace.id) });
  }

  function toggleSpaceMember(agentName) {
    if (!selectedSpace) return;
    const nextSpaces = spaces.map((space) => {
      if (space.id !== selectedSpace.id) return space;
      const members = new Set(space.members || []);
      if (members.has(agentName)) members.delete(agentName);
      else members.add(agentName);
      return { ...space, members: [...members] };
    });
    saveCommunication({ spaces: nextSpaces });
  }

  function addAgentsToSelectedSpace(names) {
    if (!selectedSpace || !names.length) return;
    const nextSpaces = spaces.map((space) => {
      if (space.id !== selectedSpace.id) return space;
      return { ...space, members: [...new Set([...(space.members || []), ...names])] };
    });
    saveCommunication({ spaces: nextSpaces });
  }

  function graphPoint(event) {
    const rect = (graphCanvasRef.current || event.currentTarget).getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function screenToGraph(point) {
    return { x: point.x - panOffset.x, y: point.y - panOffset.y };
  }

  function startCanvasGesture(event) {
    if (event.button !== 0 || event.target.closest?.(".graph-agent")) return;
    const point = graphPoint(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    if (event.shiftKey) {
      const graph = screenToGraph(point);
      setSelectionBox({ x1: graph.x, y1: graph.y, x2: graph.x, y2: graph.y });
      setGraphGesture({ mode: "selection", pointerId: event.pointerId });
      return;
    }
    setGraphGesture({
      mode: "pan",
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: panOffset.x,
      originY: panOffset.y,
    });
  }

  function moveCanvasGesture(event) {
    if (dragging) {
      moveDrag(event);
      return;
    }
    if (!graphGesture) return;
    if (graphGesture.mode === "pan") {
      setPanOffset({
        x: graphGesture.originX + event.clientX - graphGesture.startX,
        y: graphGesture.originY + event.clientY - graphGesture.startY,
      });
      return;
    }
    if (graphGesture.mode === "selection" && selectionBox) {
      const graph = screenToGraph(graphPoint(event));
      setSelectionBox((box) => ({ ...box, x2: graph.x, y2: graph.y }));
    }
  }

  function finishCanvasSelection() {
    if (!selectionBox) return;
    const left = Math.min(selectionBox.x1, selectionBox.x2);
    const right = Math.max(selectionBox.x1, selectionBox.x2);
    const top = Math.min(selectionBox.y1, selectionBox.y2);
    const bottom = Math.max(selectionBox.y1, selectionBox.y2);
    const selected = agents
      .filter((agent, index) => {
        const pos = agentPosition(agent, index, agentPositions);
        const center = { x: pos.x + 86, y: pos.y + 48 };
        return center.x >= left && center.x <= right && center.y >= top && center.y <= bottom;
      })
      .map((agent) => agent.agent_name);
    setSelectionBox(null);
    addAgentsToSelectedSpace(selected);
  }

  function finishCanvasGesture() {
    if (graphGesture?.mode === "selection") {
      finishCanvasSelection();
    } else {
      setSelectionBox(null);
    }
    setGraphGesture(null);
    finishDrag();
  }

  function startDrag(event, agentName, position) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    selectGraphAgent(agentName);
    const graph = screenToGraph(graphPoint(event));
    setDragging({
      agentName,
      pointerId: event.pointerId,
      dx: graph.x - position.x,
      dy: graph.y - position.y,
    });
  }

  function moveDrag(event) {
    if (!dragging) return;
    const graph = screenToGraph(graphPoint(event));
    const nextPositions = {
      ...agentPositions,
      [dragging.agentName]: {
        x: Math.max(20, graph.x - dragging.dx),
        y: Math.max(20, graph.y - dragging.dy),
      },
    };
    latestPositionsRef.current = nextPositions;
    setAgentPositions(nextPositions);
  }

  function finishDrag() {
    if (!dragging) return;
    setDragging(null);
    saveUi(latestPositionsRef.current, chatSessions).catch(() => {});
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <header className="brand">
          <div>
            <h1>LANGVIDEO</h1>
            <p>Agent Console</p>
          </div>
          <span className={online ? "status online" : "status offline"}>
            {online ? <CheckCircle2 size={16} /> : <TriangleAlert size={16} />}
            {online ? "online" : "offline"}
          </span>
        </header>

        <label>MainServer</label>
        <div className="inline-row">
          <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
          <button className="icon-button" onClick={refreshAgents} title="刷新">
            <RefreshCw size={18} />
          </button>
        </div>

        <form className="create-agent" onSubmit={createAgent}>
          <label>新建 Agent</label>
          <input value={newAgentName} onChange={(event) => setNewAgentName(event.target.value)} placeholder="NewAgent" />
          <div className="inline-row">
            <select value={sourceAgent} onChange={(event) => setSourceAgent(event.target.value)}>
              {sourceOptions.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <button type="submit">
              <Plus size={16} />
              新建
            </button>
          </div>
        </form>

        <div className="agent-list">
          {agents.map((agent) => (
            <button
              key={agent.agent_name}
              className={agent.agent_name === selectedAgent ? "agent-row active" : "agent-row"}
              onClick={() => selectAgent(agent.agent_name)}
            >
              <span>
                <strong>{agent.agent_name}</strong>
                <small>{agent.phase || agent.status}</small>
              </span>
              <span className="agent-state">{agent.status}</span>
            </button>
          ))}
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p>{selectedSummary?.metadata?.service_url || selectedSummary?.workspace || status}</p>
            <h2>{selectedAgent || "选择一个 Agent"}</h2>
          </div>
          <div className="actions">
            <button onClick={startAgentService} disabled={!selectedAgent || selectedServiceReady}>
              <RefreshCw size={16} />
              启动服务
            </button>
            <button onClick={stopAgentService} disabled={!selectedAgent || !selectedRegistered}>
              <Square size={16} />
              停止服务
            </button>
            <button onClick={clearRuntime} disabled={!selectedAgent}>
              <RotateCcw size={16} />
              清缓存
            </button>
            <button className="danger" onClick={deleteAgent} disabled={!selectedAgent}>
              <Trash2 size={16} />
              删除
            </button>
          </div>
        </header>

        <nav className="tabs">
          <button className={activeTab === "graph" ? "active" : ""} onClick={() => setActiveTab("graph")}>
            <Network size={16} />
            通讯图
          </button>
          <button className={activeTab === "chat" ? "active" : ""} onClick={() => setActiveTab("chat")}>
            <MessageSquare size={16} />
            对话
          </button>
          <button className={activeTab === "prompt" ? "active" : ""} onClick={() => setActiveTab("prompt")}>
            <Bot size={16} />
            提示词
          </button>
          <button className={activeTab === "monitor" ? "active" : ""} onClick={() => setActiveTab("monitor")}>
            <Activity size={16} />
            监视
          </button>
          <button className={activeTab === "config" ? "active" : ""} onClick={() => setActiveTab("config")}>
            <Settings size={16} />
            配置
          </button>
        </nav>

        {activeTab === "graph" && (
          <section className="graph-view">
            <div
              ref={graphCanvasRef}
              className={graphGesture?.mode === "pan" ? "graph-canvas panning" : "graph-canvas"}
              onPointerDown={startCanvasGesture}
              onPointerMove={moveCanvasGesture}
              onPointerUp={finishCanvasGesture}
              onPointerCancel={finishCanvasGesture}
            >
              <div className="graph-world" style={{ transform: `translate(${panOffset.x}px, ${panOffset.y}px)` }}>
                <svg className="graph-lines">
                  {edges.map((edge) => {
                    const leftAgent = agents.find((agent) => agent.agent_name === edge.from);
                    const rightAgent = agents.find((agent) => agent.agent_name === edge.to);
                    if (!leftAgent || !rightAgent) return null;
                    const leftIndex = agents.indexOf(leftAgent);
                    const rightIndex = agents.indexOf(rightAgent);
                    const leftPos = agentPosition(leftAgent, leftIndex, agentPositions);
                    const rightPos = agentPosition(rightAgent, rightIndex, agentPositions);
                    return (
                      <line
                        key={`${edge.from}-${edge.to}`}
                        x1={leftPos.x + 86}
                        y1={leftPos.y + 48}
                        x2={rightPos.x + 86}
                        y2={rightPos.y + 48}
                      />
                    );
                  })}
                </svg>
                {agents.map((agent, index) => {
                  const position = agentPosition(agent, index, agentPositions);
                  return (
                    <button
                      key={agent.agent_name}
                      className={agent.agent_name === selectedAgent ? "graph-agent active" : "graph-agent"}
                      style={{ left: position.x, top: position.y }}
                      onPointerDown={(event) => startDrag(event, agent.agent_name, position)}
                      onClick={() => selectGraphAgent(agent.agent_name)}
                      onDoubleClick={() => selectAgent(agent.agent_name, "chat")}
                      title="Agent"
                    >
                      <strong>{agent.agent_name}</strong>
                      <span>{agent.status}</span>
                      <small>{(agent.communication_spaces || []).join(", ") || "no space"}</small>
                    </button>
                  );
                })}
                {selectionBox && (
                  <div
                    className="selection-box"
                    style={{
                      left: Math.min(selectionBox.x1, selectionBox.x2),
                      top: Math.min(selectionBox.y1, selectionBox.y2),
                      width: Math.abs(selectionBox.x1 - selectionBox.x2),
                      height: Math.abs(selectionBox.y1 - selectionBox.y2),
                    }}
                  />
                )}
              </div>
            </div>
            <aside className="space-panel">
              <h3>对话空间</h3>
              <div className="inline-row">
                <input value={newSpaceName} onChange={(event) => setNewSpaceName(event.target.value)} placeholder="新空间" />
                <button className="icon-button" onClick={addSpace} title="新增空间">
                  <Plus size={16} />
                </button>
              </div>
              <select value={selectedSpaceId} onChange={(event) => setSelectedSpaceId(event.target.value)}>
                <option value="">选择空间</option>
                {spaces.map((space) => (
                  <option key={space.id} value={space.id}>
                    {space.name}
                  </option>
                ))}
              </select>
              <button className="danger" onClick={deleteSelectedSpace} disabled={!selectedSpace}>
                <Trash2 size={16} />
                删除空间
              </button>
              {selectedSpace ? (
                <div className="member-list">
                  {agents.map((agent) => (
                    <button
                      key={agent.agent_name}
                      className={(selectedSpace.members || []).includes(agent.agent_name) ? "chip active" : "chip"}
                      onClick={() => toggleSpaceMember(agent.agent_name)}
                    >
                      {agent.agent_name}
                    </button>
                  ))}
                </div>
              ) : (
                <p className="note">先创建或选择一个空间。框选图上的 Agent 会加入当前空间。</p>
              )}
              <p className="note">同一空间内成员可以互发邮件；多个空间可重叠，共享成员像维恩图交集一样分别连到两边。</p>
            </aside>
          </section>
        )}

        {activeTab === "chat" && (
          <section className="chat-view">
            <div className="conversation">
              <div className="messages" ref={messagesRef}>
                {chatMessages.map((message, index) => {
                  const roleClass = String(message.role || "").replace(/\s+/g, "-");
                  const visibleEvents = (message.events || []).filter(isVisibleTraceEvent);
                  return (
                    <div key={message.id || `${message.role}-${index}`} className={`bubble ${roleClass}`}>
                      <div>{message.content}</div>
                      {visibleEvents.length > 0 && (
                        <div className="bubble-trace">
                          {visibleEvents.slice(0, 8).map((eventItem) => (
                            <p key={eventItem.id || `${eventItem.title}-${eventItem.detail}`}>
                              <strong>{eventItem.title}</strong>
                              {eventItem.detail ? <span>{eventItem.detail}</span> : null}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
                {selectedAgentActivity.length > 0 && (
                  <div className="activity-stream">
                    <div className="activity-heading">Agent 行为流</div>
                    {selectedAgentActivity.map((message) => (
                      message.role === "agent" ? (
                        <div key={message.id} className="bubble agent">
                          {message.content}
                        </div>
                      ) : (
                        <div key={message.id} className={`bubble behavior ${message.kind || "tool"}`}>
                          <div className="behavior-title">
                            <span>{message.kind === "tool" ? "工具" : "消息"}</span>
                            <strong>{message.content}</strong>
                          </div>
                          {Array.isArray(message.events) && message.events.length > 0 && (
                            <div className="bubble-trace">
                              {message.events.map((eventItem) => (
                                <p key={eventItem.id || `${eventItem.title}-${eventItem.detail}`}>
                                  <strong>{eventItem.title}</strong>
                                  {eventItem.detail ? <span>{eventItem.detail}</span> : null}
                                </p>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    ))}
                  </div>
                )}
              </div>
              <form className="composer" onSubmit={sendChat}>
                <textarea value={chatText} onChange={(event) => setChatText(event.target.value)} placeholder="输入消息" />
                <input
                  value={attachmentUrl}
                  onChange={(event) => setAttachmentUrl(event.target.value)}
                  placeholder="图片 URL、data URL 或文件链接"
                />
                <div className="composer-actions">
                  <select value={sendMode} onChange={(event) => setSendMode(event.target.value)}>
                    <option value="direct">正常发送</option>
                    <option value="mail">发邮件</option>
                  </select>
                  <button type="submit">
                    <Send size={16} />
                    发送
                  </button>
                </div>
              </form>
            </div>
            <aside className="runtime-panel">
              <h3>对话配置</h3>
              <label>Thread ID</label>
              <input value={threadId} onChange={(event) => setThreadId(event.target.value)} />
              <label>Run ID</label>
              <input value={runId} onChange={(event) => setRunId(event.target.value)} />
              <label>Stream Mode JSON</label>
              <textarea value={streamMode} onChange={(event) => setStreamMode(event.target.value)} />
              <button onClick={saveChatConfig}>
                <Save size={16} />
                保存配置
              </button>
              <button onClick={newThread}>
                <Plus size={16} />
                新线程
              </button>
              {!selectedServiceReady && <p className="warning-note">当前 AgentServer 未注册。direct 对话前请先启动服务。</p>}
              <div className="mini-monitor">
                <h4>运行</h4>
                {monitor.agents?.map((agent) => (
                  <p key={agent.agent_name}>
                    <strong>{agent.agent_name}</strong> {agent.status} {agent.phase || ""}
                  </p>
                ))}
              </div>
              <div className="trace-panel">
                <h4>Agent 行为</h4>
                {selectedAgentActivity.length ? (
                  selectedAgentActivity.slice(-12).map((eventItem) => (
                    <div key={eventItem.id} className={`trace-row behavior ${eventItem.kind || "message"}`}>
                      <strong>{eventItem.kind === "tool" ? "工具" : "消息"} · {eventItem.role === "agent" ? "Agent 回复" : eventItem.content}</strong>
                      <p>{eventItem.role === "agent" ? eventItem.content : eventItem.events?.[0]?.detail || ""}</p>
                    </div>
                  ))
                ) : lastTraceEvents.filter(isVisibleTraceEvent).length ? (
                  lastTraceEvents.filter(isVisibleTraceEvent).map((eventItem) => (
                    <div key={eventItem.id || `${eventItem.title}-${eventItem.detail}`} className={`trace-row ${eventItem.type}`}>
                      <strong>{eventItem.title}</strong>
                      {eventItem.detail ? <p>{eventItem.detail}</p> : null}
                    </div>
                  ))
                ) : (
                  <p className="note">该 agent 的消息和工具调用会显示在这里。</p>
                )}
              </div>
            </aside>
          </section>
        )}

        {activeTab === "prompt" && (
          <section className="prompt-view prompt-split">
            <div>
              <div className="form-grid">
                <label>Role</label>
                <input value={agentRole} onChange={(event) => setAgentRole(event.target.value)} />
                <label>Description</label>
                <textarea value={agentDescription} onChange={(event) => setAgentDescription(event.target.value)} />
                <label>Responsibilities</label>
                <textarea className="tall" value={agentResponsibilities} onChange={(event) => setAgentResponsibilities(event.target.value)} />
                <label>System Extra</label>
                <textarea className="tall" value={systemPromptExtra} onChange={(event) => setSystemPromptExtra(event.target.value)} />
                <label>Default Run ID</label>
                <input value={defaultRunId} onChange={(event) => setDefaultRunId(event.target.value)} />
                <label>Knowledge Run ID</label>
                <input value={knowledgeRunId} onChange={(event) => setKnowledgeRunId(event.target.value)} />
              </div>
              <button onClick={savePrompt}>
                <Save size={16} />
                保存运行提示词
              </button>
            </div>
            <div>
              <label>Brain AGENTS.md</label>
              <textarea className="code brain-editor" value={brainPrompt} onChange={(event) => setBrainPrompt(event.target.value)} />
              <button onClick={saveBrainPrompt}>
                <Save size={16} />
                保存 Brain
              </button>
            </div>
          </section>
        )}

        {activeTab === "monitor" && (
          <section className="monitor-view">
            <div>
              <h3>Agent 状态</h3>
              {monitor.agents?.map((agent) => (
                <div key={agent.agent_name} className="monitor-row">
                  <strong>{agent.agent_name}</strong>
                  <span>{agent.status}</span>
                  <span>{agent.phase || "-"}</span>
                  <small>{agent.step || ""}</small>
                </div>
              ))}
            </div>
            <div>
              <h3>邮件</h3>
              {(monitor.recent_mail || []).slice().reverse().map((item, index) => (
                <div key={`${item.at}-${index}`} className="mail-row">
                  <strong>
                    {item.message?.from} {"->"} {item.message?.to}
                  </strong>
                  <span>{item.message?.type}</span>
                  <p>{typeof item.message?.content === "string" ? item.message.content : pretty(item.message?.content)}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {activeTab === "config" && (
          <section className="split-view">
            <div>
              <label>MainServer Config</label>
              <textarea className="code" value={mainConfigJson} onChange={(event) => setMainConfigJson(event.target.value)} />
              <button onClick={saveMainConfig}>
                <Database size={16} />
                保存中心配置
              </button>
            </div>
            <div>
              <label>Agent Runtime Config</label>
              <textarea className="code" value={runtimeConfigJson} onChange={(event) => setRuntimeConfigJson(event.target.value)} />
              <button onClick={saveRuntimeConfig}>
                <Database size={16} />
                保存运行配置
              </button>
            </div>
          </section>
        )}

        <footer className="footer">
          <span>{busy ? "Working..." : status}</span>
        </footer>
      </main>
    </div>
  );
}

const rootElement = document.getElementById("root");
globalThis.__langvideoRoot = globalThis.__langvideoRoot || createRoot(rootElement);
globalThis.__langvideoRoot.render(<App />);
