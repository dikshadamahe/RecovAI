# RecovAI — one-shot Vercel deploy (Neon Postgres + env + production)
# Run from repo root:  powershell -ExecutionPolicy Bypass -File .\scripts\deploy-vercel.ps1

$ErrorActionPreference = "Stop"
# Vercel CLI writes its banner to stderr; do not treat that as a terminating error.
$PrevEap = $ErrorActionPreference
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Team = "team_omHPbwlRoEXwxHRbGtM5Rgxr"
$Project = "recovai-backend"

Write-Host "==> Checking Vercel CLI login..."
$Vercel = "vercel.cmd"
$ErrorActionPreference = "Continue"
$who = & $Vercel whoami 2>&1 | Out-String
$ErrorActionPreference = $PrevEap
if ($LASTEXITCODE -ne 0 -or $who -notmatch '\S') {
    Write-Host $who
    Write-Host "Login failed. Run: vercel login"
    exit 1
}
Write-Host "Logged in as: $($who.Trim())"

Write-Host "==> Linking project $Project..."
& $Vercel link --yes --project $Project --scope $Team

Write-Host "==> Provisioning Neon Postgres (free plan) and connecting to project..."
# Non-interactive; creates DATABASE_URL on the project when successful.
& $Vercel integration add neon --name recovai-db -e production -e preview --json 2>&1 | Write-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "Neon install failed (maybe already exists). Continue if DATABASE_URL is set in Vercel dashboard."
}

if (Test-Path ".env") {
    $groq = (Get-Content ".env" | Where-Object { $_ -match '^GROQ_API_KEY=' }) -replace '^GROQ_API_KEY=', ''
    if ($groq) {
        Write-Host "==> Setting GROQ_API_KEY on Vercel..."
        $groq | & $Vercel env add GROQ_API_KEY production --scope $Team --force 2>&1 | Out-Null
        $groq | & $Vercel env add GROQ_API_KEY preview --scope $Team --force 2>&1 | Out-Null
    }
}

Write-Host "==> Setting VERCEL_SUPPORT_LARGE_FUNCTIONS=1..."
"1" | & $Vercel env add VERCEL_SUPPORT_LARGE_FUNCTIONS production --scope $Team --force 2>&1 | Out-Null
"1" | & $Vercel env add VERCEL_SUPPORT_LARGE_FUNCTIONS preview --scope $Team --force 2>&1 | Out-Null

Write-Host "==> Deploying to production..."
& $Vercel deploy --prod --yes --scope $Team
Write-Host "Done. Open the production URL shown above and test /health and /api/test."
