@direction LR
@spacing 80

[User Query] -> [AgentCoordinator]
[AgentCoordinator] -> [Coordinator Intelligence Layer
EvidenceGraph]
[Coordinator Intelligence Layer
EvidenceGraph] -> [Coordinator Output
schema_version 2.1]
[Coordinator Output
schema_version 2.1] -> [Signal Studio Readout / Proof / Monitor]
[Coordinator Output
schema_version 2.1] -> [ReportEngine Adapter]
[ReportEngine Adapter] -> [Chapter JSON]
[Chapter JSON] -> [Document IR]
[Document IR] -> [HTML]
[Document IR] -> [Markdown]
[Document IR] -> [PDF]
[Document IR] -> [State / Manifest Files]
