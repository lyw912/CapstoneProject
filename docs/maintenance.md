# Documentation Maintenance

Use this checklist when changing code, APIs, diagrams, or runtime assets.

## Local Documentation Checks

| Check | Command Or Method |
| --- | --- |
| Search for non-English public docs | `Select-String -Path README.md,docs/**/*.md -Pattern '[\u4e00-\u9fff]'` |
| Search for stale maintenance markers | `rg -n "T[O]DO|TBD|FIXME|placeholder|WIP" README.md docs -i` |
| Check local links | Use the PowerShell script below or a Markdown link checker. |
| Check OpenAPI syntax | Parse `docs/reference/openapi.yaml` with a YAML parser. |
| Check API route drift | Compare Flask route decorators with `docs/reference/api.md` and `openapi.yaml`. |
| Run Python checks without `python` on PATH | Use `uv run --python 3.11 ...` or activate the Conda environment. |

PowerShell link check:

```powershell
$root=(Get-Location).Path
$files=@('README.md') + (Get-ChildItem docs -Recurse -File -Filter *.md | ForEach-Object { $_.FullName.Substring($root.Length+1) })
foreach ($rel in $files) {
  $text=Get-Content -Raw $rel
  $dir=Split-Path $rel
  if ($dir -eq '') { $dir='.' }
  [regex]::Matches($text, '!\[[^\]]*\]\(([^)]+)\)|(?<!! )\[[^\]]+\]\(([^)]+)\)') | ForEach-Object {
    $target=$_.Groups[1].Value
    if (-not $target) { $target=$_.Groups[2].Value }
    if ($target -match '^(https?:|mailto:|#)') { return }
    $clean=($target -split '#')[0].Trim('<','>')
    if ($clean -and -not (Test-Path (Join-Path $dir $clean))) {
      "BROKEN $rel -> $target"
    }
  }
}
```

## Update Rules

| If You Change | Also Update |
| --- | --- |
| Flask route behavior | `docs/reference/api.md`, `docs/reference/openapi.yaml`, tests. |
| Coordinator output fields | `docs/reference/coordinator-output-schema.md`, Signal Studio consumers, bridge tests. |
| Report IR block types | `docs/reference/report-ir.md`, validators, renderer tests. |
| Provider defaults | `.env.example`, `docs/reference/configuration.md`, `docs/quality/api-evaluation.md`. |
| UI views | `docs/components/signal-studio.md`, screenshots, acceptance walkthrough. |
| Deployment shape | `Dockerfile`, `docker-compose.yml`, `docs/operations/deployment.md`. |
| Runtime asset paths | `docs/reference/runtime-assets.md`, `docs/reference/data-and-model-assets.md`. |

## CI Recommendation

Keep the GitHub Actions workflow lightweight:

| Job | Purpose |
| --- | --- |
| Markdown link check | Prevent broken relative links in README/docs. |
| OpenAPI parse check | Prevent invalid YAML and schema-field drift. |
| Focused tests | Run the documented backend regression suite when Python dependencies are available. |

The existing Docker workflow publishes images on version tags. Documentation checks should run on pull requests and pushes to main branches.

## Python Execution Policy

All Python commands in this project should run on Python 3.11 or newer. Acceptable execution paths are:

| Path | Example |
| --- | --- |
| Activated venv/system Python | `python -m unittest tests.test_sensitive_input_filter` |
| Conda | `conda activate capstone-project` then `python -m unittest tests.test_sensitive_input_filter` |
| `uv` managed execution | `uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_sensitive_input_filter` |

Do not downgrade test or runtime commands below Python 3.11 to work around environment issues.
