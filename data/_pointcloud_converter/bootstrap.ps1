param([Parameter(Mandatory = $true)][string]$DataRoot)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if (-not [Environment]::Is64BitOperatingSystem) {
    Write-Host 'This converter requires 64-bit Windows 10 or Windows 11.'
    exit 1
}
$cacheBase = $env:LOCALAPPDATA
if ([string]::IsNullOrWhiteSpace($cacheBase)) { $cacheBase = $env:TEMP }
$runtime = Join-Path $cacheBase 'MCBPointcloudConverter\runtime-v1'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$env:UV_CACHE_DIR = Join-Path $runtime 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$uv = Join-Path $runtime 'uv.exe'
$uvHashFile = Join-Path $runtime 'uv.sha256'
$envLock = $null
try {
    $envLock = [IO.File]::Open((Join-Path $runtime 'environment.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $cachedUvValid = $false
    if ((Test-Path $uv) -and (Test-Path $uvHashFile)) {
        $cachedUvValid = (Get-FileHash $uv -Algorithm SHA256).Hash -eq (Get-Content $uvHashFile -Raw).Trim()
    }
    if (-not $cachedUvValid) {
        Write-Host 'Downloading the pinned uv runtime from the official release...'
        $version = '0.13.0'
        $assetName = 'uv-x86_64-pc-windows-msvc.zip'
        # SHA-256 of the official uv 0.13.0 Windows x64 release asset.
        $expectedHash = '088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c'
        $zip = Join-Path $runtime 'uv-download.zip'
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$version/$assetName" -OutFile $zip -TimeoutSec 1800
        if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne $expectedHash) {
            throw 'uv download checksum mismatch. Please retry.'
        }
        $unpack = Join-Path $runtime 'uv-unpack'
        if (Test-Path $unpack) { Remove-Item $unpack -Recurse -Force }
        Expand-Archive -LiteralPath $zip -DestinationPath $unpack -Force
        $exe = Get-ChildItem -LiteralPath $unpack -Filter 'uv.exe' -File -Recurse | Select-Object -First 1
        if ($null -eq $exe) { throw 'uv.exe is missing from the downloaded archive.' }
        Copy-Item -LiteralPath $exe.FullName -Destination $uv -Force
        (Get-FileHash $uv -Algorithm SHA256).Hash | Set-Content -LiteralPath $uvHashFile -Encoding ASCII
        Remove-Item $zip -Force
        Remove-Item $unpack -Recurse -Force
    }
    $pythonRequest = 'cpython-3.12.15-windows-x86_64-none'
    Write-Host 'Preparing isolated Python 3.12.15 (no PATH or registry changes)...'
    & $uv --no-config --system-certs python install $pythonRequest --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw 'Python download failed. Check internet access and retry.' }
    $venv = Join-Path $runtime 'venv'
    $python = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $python)) {
        & $uv --no-config --system-certs venv $venv --python $pythonRequest --managed-python
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python environment.' }
    }
    & $python -c "import importlib.util, sys; s=importlib.util.find_spec('numpy'); sys.exit(0 if s is not None and __import__('numpy').__version__ == '2.5.4' else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing NumPy 2.5.4...'
        & $uv --no-config --system-certs pip install --python $python 'numpy==2.5.4'
        if ($LASTEXITCODE -ne 0) { throw 'NumPy installation failed. Check internet access and retry.' }
    }
    $envLock.Dispose()
    $envLock = $null
    $source = Join-Path $DataRoot 'MCB_merged'
    $output = Join-Path $DataRoot 'MCB_pointcloud'
    $script = Join-Path $DataRoot '_pointcloud_converter\convert_mcb.py'
    $workers = '0'
    if (-not [string]::IsNullOrWhiteSpace($env:MCB_WORKERS)) { $workers = $env:MCB_WORKERS }
    & $python -u $script --source $source --output $output --workers $workers
    exit $LASTEXITCODE
} catch {
    Write-Host ''
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Runtime cache: ' + $runtime)
    exit 1
} finally {
    if ($null -ne $envLock) { $envLock.Dispose() }
}
