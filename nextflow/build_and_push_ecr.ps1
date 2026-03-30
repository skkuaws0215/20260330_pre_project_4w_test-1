param(
    [string]$ImageTag = "fe-latest",
    [string]$RepositoryUri = "666803869796.dkr.ecr.ap-northeast-2.amazonaws.com/skku-project/pre-4team",
    [string]$Region = "ap-northeast-2"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Team AWS profile"
. .\use-team-aws.ps1

Write-Host "[2/4] ECR login"
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $RepositoryUri

Write-Host "[3/4] Docker build"
docker build -t "${RepositoryUri}:${ImageTag}" -f nextflow/Dockerfile .

Write-Host "[4/4] Docker push"
docker push "${RepositoryUri}:${ImageTag}"

Write-Host "Pushed: ${RepositoryUri}:${ImageTag}"
