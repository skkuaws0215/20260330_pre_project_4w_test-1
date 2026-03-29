# 개인 계정: 기본값 %USERPROFILE%\.aws (건드리지 않음)
# 팀 계정: 이 스크립트가 현재 세션에서만 config/credentials 경로를 프로젝트 .aws 로 지정
$ProjectRoot = $PSScriptRoot
$env:AWS_CONFIG_FILE = Join-Path $ProjectRoot ".aws\config"
$env:AWS_SHARED_CREDENTIALS_FILE = Join-Path $ProjectRoot ".aws\credentials"
$env:AWS_PROFILE = "4team-project"

Write-Host "AWS_CONFIG_FILE=$($env:AWS_CONFIG_FILE)"
Write-Host "AWS_SHARED_CREDENTIALS_FILE=$($env:AWS_SHARED_CREDENTIALS_FILE)"
Write-Host "AWS_PROFILE=$($env:AWS_PROFILE)"
Write-Host "예: aws sts get-caller-identity"
