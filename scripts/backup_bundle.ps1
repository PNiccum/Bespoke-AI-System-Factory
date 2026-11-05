<# Backs up the repo as a single git bundle file into Dropbox. Run by hand or from Task Scheduler.
   Usage: powershell -File scripts\backup_bundle.ps1 [-Repo C:\path\to\repo] [-Dest "<dropbox folder>"] [-Keep 14]
   A bundle is one file holding all branches and history; restore with: git clone <file>.bundle restored
   Never keep the live repo inside Dropbox (sync can corrupt .git). Contains committed history only. #>
param(
  [string]$Repo = (Resolve-Path "$PSScriptRoot\..").Path,
  [string]$Dest = "$env:USERPROFILE\Dropbox\School Stuff\ADS\Agentic AI in Practice\Final Project\repo-backups",
  [int]$Keep = 14
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out = Join-Path $Dest "aspire-$stamp.bundle"
git -C $Repo bundle create $out --all
git bundle verify $out
Get-ChildItem $Dest -Filter "aspire-*.bundle" | Sort-Object LastWriteTime -Descending |
  Select-Object -Skip $Keep | Remove-Item
Write-Host "Backup written: $out"
