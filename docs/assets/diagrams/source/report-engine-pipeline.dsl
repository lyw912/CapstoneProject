@direction TB
@spacing 70

[Coordinator Artifact\nor legacy engine files] -> [Normalize Inputs]
[Normalize Inputs] -> [Template Selection]
[Template Selection] -> [Template Slice]
[Template Slice] -> [Document Layout]
[Document Layout] -> [Word Budget]
[Word Budget] -> [Prepare Storage]
[Prepare Storage] -> [Process Chapter]
[Process Chapter] -> [Process Chapter]
[Process Chapter] -> [Finalize Report]
[Finalize Report] -> [Document IR]
[Document IR] -> [HTML Renderer]
[Document IR] -> [Markdown Renderer]
[Document IR] -> [PDF Renderer]
