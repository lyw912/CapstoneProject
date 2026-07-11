@direction TB
@spacing 70

[Start Runtime
/api/system/start] -> [Stop legacy Streamlit apps]
[Stop legacy Streamlit apps] -> [Stop Forum monitor
for Signal Studio]
[Stop Forum monitor
for Signal Studio] -> [Initialize ReportEngine]
[Initialize ReportEngine] -> [Runtime ready]
[Runtime ready] -> [Run analysis
/api/coordinator/run]
[Run analysis
/api/coordinator/run] -> [Coordinator Intelligence Layer
EvidenceGraph + provider diagnostics]
[Coordinator Intelligence Layer
EvidenceGraph + provider diagnostics] -> [Poll coordinator task
/api/coordinator/task/{id}]
[Poll coordinator task
/api/coordinator/task/{id}] -> [Write coordinator_output_latest.json
schema 2.1]
[Write coordinator_output_latest.json
schema 2.1] -> [Load latest artifact
/api/coordinator/latest]
[Load latest artifact
/api/coordinator/latest] -> [Generate report
/api/report/generate]
[Generate report
/api/report/generate] -> [Stream report events
/api/report/stream/{id}]
[Stream report events
/api/report/stream/{id}] -> [Edit / render Document IR
/api/report/render-ir]
[Edit / render Document IR
/api/report/render-ir] -> [Export HTML / Markdown / PDF]
