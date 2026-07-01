@direction LR
@spacing 80

[User Query] -> [AgentCoordinator]
[AgentCoordinator] -> [Coordinator Output\nschema_version 1.0]
[Coordinator Output\nschema_version 1.0] -> [Signal Studio Readout]
[Coordinator Output\nschema_version 1.0] -> [ReportEngine Adapter]
[ReportEngine Adapter] -> [Chapter JSON]
[Chapter JSON] -> [Document IR]
[Document IR] -> [HTML]
[Document IR] -> [Markdown]
[Document IR] -> [PDF]
[Document IR] -> [State / Manifest Files]
