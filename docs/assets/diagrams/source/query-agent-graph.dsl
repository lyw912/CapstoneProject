@direction LR
@spacing 80

(START) -> [Query Planner]
[Query Planner] -> [Unified Search]
[Unified Search] -> [Dedup Filter]
[Dedup Filter] -> [Trust Scorer]
[Trust Scorer] -> [Stance Classifier]
[Stance Classifier] -> [Social Enrichment]
[Social Enrichment] -> [Coverage Check]
[Coverage Check] -> [Gap Filler]
[Gap Filler] -> [Unified Search]
[Coverage Check] -> [Output Assembly]
[Output Assembly] -> (END)
