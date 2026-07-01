@direction TB
@spacing 70

(START) -> [Query Agent]
(START) -> [Media Agent]
[Query Agent] -> [Data Bridge]
[Media Agent] -> [Data Bridge]
[Data Bridge] -> [Divergence Matrix]
[Divergence Matrix] -> [Perspective Generator]
[Perspective Generator] -> [Deliberation Engine]
[Deliberation Engine] -> [Gap Detector]
[Gap Detector] -> [Targeted Search]
[Targeted Search] -> [Deliberation Engine]
[Gap Detector] -> [Echo Chamber Detector]
[Echo Chamber Detector] -> [Fact / Opinion Separator]
[Fact / Opinion Separator] -> [Platform Interpreter]
[Platform Interpreter] -> [Synthesis]
[Synthesis] -> [Report Agent Node]
[Report Agent Node] -> (END)
