@direction TB
@spacing 70

[Start Runtime\n/api/system/start] -> [Stop legacy Streamlit apps]
[Stop legacy Streamlit apps] -> [Stop Forum monitor\nfor Signal Studio]
[Stop Forum monitor\nfor Signal Studio] -> [Initialize ReportEngine]
[Initialize ReportEngine] -> [Runtime ready]
[Runtime ready] -> [Run analysis\n/api/coordinator/run]
[Run analysis\n/api/coordinator/run] -> [Poll coordinator task\n/api/coordinator/task/{id}]
[Poll coordinator task\n/api/coordinator/task/{id}] -> [Write coordinator_output_latest.json]
[Write coordinator_output_latest.json] -> [Load latest artifact\n/api/coordinator/latest]
[Load latest artifact\n/api/coordinator/latest] -> [Generate report\n/api/report/generate]
[Generate report\n/api/report/generate] -> [Stream report events\n/api/report/stream/{id}]
[Stream report events\n/api/report/stream/{id}] -> [Export HTML / Markdown / PDF]
