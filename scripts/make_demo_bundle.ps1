# Builds dist\FetalAI-demo.zip: the code (git-tracked files), the trained models (the .pth files are not in git)
# and a demo_samples folder with dataset images that each give a different result. Copy the zip to another
# laptop, unzip, double-click start_fetalai.bat.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\make_demo_bundle.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stage = Join-Path $Root "dist\FetalAI-demo"
$Zip = Join-Path $Root "dist\FetalAI-demo.zip"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
if (Test-Path $Zip) { Remove-Item -Force $Zip }
New-Item -ItemType Directory -Force $Stage | Out-Null

Write-Host "Copying git-tracked files"
foreach ($f in (git ls-files)) {
    $dst = Join-Path $Stage $f
    New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
    Copy-Item -LiteralPath $f -Destination $dst
}

Write-Host "Copying trained models"
foreach ($m in @("outputs\hc18_unet_best.pth", "outputs\plane_classifier_best.pth")) {
    if (-not (Test-Path $m)) { throw "Missing $m" }
    Copy-Item $m (Join-Path $Stage $m)
}

Write-Host "Copying demo images"
$Samples = Join-Path $Stage "demo_samples"
New-Item -ItemType Directory -Force $Samples | Out-Null
$images = [ordered]@{
    "1_PASS_544_HC.png"              = "data\raw\HC18\training_set\544_HC.png"
    "2_WARN_withheld_006_HC.png"     = "data\raw\HC18\training_set\006_HC.png"
    "3_PASS_low_reliability_794_HC.png" = "data\raw\HC18\training_set\794_HC.png"
    "4_WARN_femur_Patient00647.png"  = "data\raw\FETAL_PLANES_DB\Images\Patient00647_Plane5_1_of_1.png"
}
foreach ($k in $images.Keys) {
    if (-not (Test-Path $images[$k])) { throw "Missing $($images[$k])" }
    Copy-Item $images[$k] (Join-Path $Samples $k)
}
Copy-Item "docs\demo_samples_README.txt" (Join-Path $Samples "README.txt")

Write-Host "Zipping"
Compress-Archive -Path $Stage -DestinationPath $Zip -CompressionLevel Optimal
Remove-Item -Recurse -Force $Stage
$mb = [math]::Round((Get-Item $Zip).Length / 1MB, 1)
Write-Host "Done: $Zip ($mb MB)" -ForegroundColor Green
