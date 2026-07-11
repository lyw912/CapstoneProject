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
public boundary] -> [Fusion Parent Graph
plan / fan-out / audit / stopping]
[Fusion Parent Graph
plan / fan-out / audit / stopping] -> [QueryEngine Subgraph
breadth / stance / MindSpiderDB]
[Fusion Parent Graph
plan / fan-out / audit / stopping] -> [MediaEngine Subgraph
sections / reflection / multimodal]
[QueryEngine Subgraph
breadth / stance / MindSpiderDB] -> [Evidence Blackboard
canonical sources + acquisitions]
[MediaEngine Subgraph
sections / reflection / multimodal] -> [Evidence Blackboard
canonical sources + acquisitions]
[Evidence Blackboard
canonical sources + acquisitions] -> [EvidenceCore
quality / claims / edges / audit]
[EvidenceCore
quality / claims / edges / audit] -> [Coordinator Artifact
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
