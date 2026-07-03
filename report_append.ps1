<#
.SYNOPSIS
    Run a tokengraph savings report for one or more queries and APPEND the
    per-query rows to a cumulative CSV (report.csv by default).

    Unlike `tokengraph_all.py report --csv`, which overwrites the file on every
    run, this script accumulates rows across runs: the header is written only
    when the file is first created; subsequent runs append their query rows.

.PARAMETER Queries
    One or more task/query strings. Positional, so you can just list them.

.PARAMETER Csv
    Path to the cumulative CSV. Defaults to report.csv next to this script.

.PARAMETER TasksFile
    Optional file with one query per line (# comments allowed). Combined with
    any positional queries.

.PARAMETER Budget
    Token budget per query pack (passed through to the report). Default 6000.

.PARAMETER Markdown
    Also append a timestamped, human-readable markdown report section to
    report.md (or -MarkdownPath). Each run becomes its own "## Run <time>"
    block, so the file is a running log alongside the cumulative CSV.

.PARAMETER MarkdownPath
    Path to the cumulative markdown log. Defaults to report.md next to this
    script. Specifying it implies -Markdown.

.EXAMPLE
    .\report_append.ps1 "how does indexing work" "how does semantic search work"

.EXAMPLE
    .\report_append.ps1 -TasksFile queries.txt -Csv runs.csv

.EXAMPLE
    .\report_append.ps1 -Markdown "how does reindex work"
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Queries,

    [string]$Csv,

    [string]$TasksFile,

    [int]$Budget = 6000,

    [switch]$Markdown,

    [string]$MarkdownPath
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $scriptDir 'tokengraph_all.py'

if (-not $Csv) { $Csv = Join-Path $scriptDir 'report.csv' }
if (-not $MarkdownPath) { $MarkdownPath = Join-Path $scriptDir 'report.md' }
# -MarkdownPath implies -Markdown so callers can just point at a file.
if ($MarkdownPath -and $PSBoundParameters.ContainsKey('MarkdownPath')) { $Markdown = $true }

# Collect queries from positional args and/or a tasks file.
$allQueries = @()
if ($Queries) { $allQueries += $Queries }
if ($TasksFile) {
    if (-not (Test-Path $TasksFile)) { throw "TasksFile not found: $TasksFile" }
    foreach ($line in Get-Content $TasksFile) {
        $t = $line.Trim()
        if ($t -and -not $t.StartsWith('#')) { $allQueries += $t }
    }
}

if ($allQueries.Count -eq 0) {
    throw "No queries given. Pass query strings and/or -TasksFile FILE."
}

# Run the report into a fresh temp CSV (the built-in --csv overwrites, which is
# fine for this throwaway file), then append its rows to the cumulative CSV.
# The report's markdown is printed to stdout, which we capture for the optional
# markdown append.
$tmp = [System.IO.Path]::GetTempFileName()
try {
    $args = @("`"$py`"", 'report', '-b', $Budget, '--csv', "`"$tmp`"") + ($allQueries | ForEach-Object { "`"$_`"" })
    $mdOutput = & python @args
    # Drop the trailing "wrote <tmp>" confirmation line the CLI prints for --csv.
    $mdBody = ($mdOutput | Where-Object { $_ -notmatch "^wrote " }) -join "`n"

    if (-not (Test-Path $tmp)) { throw "Report did not produce a CSV." }
    $lines = Get-Content $tmp

    if (Test-Path $Csv) {
        $rows = $lines | Select-Object -Skip 1   # drop header, keep data rows
        if ($rows) { Add-Content -Path $Csv -Value $rows -Encoding utf8 }
        Write-Output "Appended $($rows.Count) row(s) to $Csv"
    }
    else {
        Set-Content -Path $Csv -Value $lines -Encoding utf8
        Write-Output "Created $Csv with header + $(($lines.Count) - 1) row(s)"
    }

    # Optional: append a timestamped, human-readable markdown block. Each run is
    # added as its own "## Run <timestamp>" section so the file is a running log.
    if ($Markdown) {
        $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        $block = @()
        if (-not (Test-Path $MarkdownPath)) {
            $block += "# Tokengraph savings report (running log)"
            $block += ""
        }
        $block += "## Run $stamp ($($allQueries.Count) quer$(if ($allQueries.Count -eq 1) {'y'} else {'ies'}))"
        $block += ""
        $block += $mdBody
        $block += ""
        Add-Content -Path $MarkdownPath -Value ($block -join "`n") -Encoding utf8
        Write-Output "Appended run section to $MarkdownPath"
    }
}
finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}
