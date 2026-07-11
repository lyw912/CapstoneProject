@direction LR
@spacing 80

[Operator] -> [Signal Studio
React / Vite UI]
[Signal Studio
React / Vite UI] -> [Flask Orchestrator
app.py]
[Flask Orchestrator
app.py] -> [AgentCoordinator
public boundary]
[AgentCoordinator
public boundary] -> [Coordinator Intelligence Layer
EvidenceGraph]
[Coordinator Intelligence Layer
EvidenceGraph] -> [Source Gateway
Tavily / Bocha / Anspire / MindSpiderDB]
[Coordinator Intelligence Layer
EvidenceGraph] -> [Quality + Claim Audit
clusters / spans / citations]
[QueryEngine
legacy/direct-use] -> [Historical evaluation + tools]
[MediaEngine
legacy/direct-use] -> [Optional historical media synthesis]
[Coordinator Intelligence Layer
EvidenceGraph] -> [Coordinator Artifact
schema 2.1 JSON cache]
[Coordinator Artifact
schema 2.1 JSON cache] -> [Signal Studio
Latest Readout]
[Coordinator Artifact
schema 2.1 JSON cache] -> [ReportEngine]
[ReportEngine] -> [Document IR]
[Document IR] -> [HTML]
[Document IR] -> [Markdown]
[Document IR] -> [PDF]
[Flask Orchestrator
app.py] -> [LangSmith
Configurable tracing]
