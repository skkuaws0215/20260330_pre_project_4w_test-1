param(
    [string]$ParamsFile = "nextflow/params/team4.awsbatch.example.json",
    [string]$JavaHome = "c:\Users\biso8\dev\tools\jdk17\extract\jdk-17.0.17+10",
    [string]$NextflowBin = "c:\Users\biso8\dev\tools\nextflow-24.10.4\nextflow",
    [string]$BashExe = "C:\Program Files\Git\bin\bash.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ParamsFile)) {
    throw "Params file not found: $ParamsFile"
}

Write-Host "[1/3] Team AWS profile"
. .\use-team-aws.ps1

if (-not (Test-Path $NextflowBin)) {
    throw "Nextflow launcher not found: $NextflowBin"
}
if (-not (Test-Path $BashExe)) {
    throw "Git Bash not found: $BashExe"
}
if (-not (Test-Path (Join-Path $JavaHome "bin\\java.exe"))) {
    throw "Java not found: $JavaHome"
}

Write-Host "[2/3] Run Nextflow on AWS Batch"
$javaPosix = $JavaHome -replace '\\','/'
$nfPosix = $NextflowBin -replace '\\','/'
$repoPosix = (Get-Location).Path -replace '\\','/'
$paramsPosix = $ParamsFile -replace '\\','/'

$bashCmd = "export JAVA_HOME=$javaPosix; export PATH=`$JAVA_HOME/bin:`$PATH; cd $repoPosix && $nfPosix run nextflow/main.nf -profile awsbatch -params-file $paramsPosix"
& $BashExe -lc $bashCmd

Write-Host "[3/3] Done. Check S3 output prefix:"
Write-Host "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/"
