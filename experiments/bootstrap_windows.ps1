param([Parameter(Mandatory = $true)][string]$ProjectRoot,
      [ValidateSet('full','quick','smoke')][string]$Profile = 'full')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
function Stop-Setup([string]$Message) { throw $Message }
try {
    if (-not [Environment]::Is64BitOperatingSystem) { Stop-Setup '64-bit Windows is required.' }
    if (-not (Test-Path (Join-Path $ProjectRoot 'PointNeXt\openpoints\models\backbone\pointnext.py'))) {
        Stop-Setup 'PointNeXt/openpoints is missing. Copy the complete project, including the submodule.'
    }
    $version = [Environment]::OSVersion.Version
    if ($version.Build -lt 19044) { Stop-Setup 'Use Windows 10 21H2 (build 19044) or newer / Windows 11 for WSL CUDA.' }
    $wsl = Join-Path $env:WINDIR 'System32\wsl.exe'
    if (-not (Test-Path $wsl)) { Stop-Setup 'wsl.exe is missing. Enable Windows Subsystem for Linux on supported Windows.' }
    $distro = 'Ubuntu-24.04'
    Write-Host 'Checking/updating the WSL runtime...'
    $ErrorActionPreference = 'Continue'
    & $wsl --update
    $updateExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($updateExit -ne 0) { Write-Host 'WSL update did not finish; installation/startup checks follow. If needed, run wsl --update as administrator.' }
    $ErrorActionPreference = 'Continue'
    $installed = & $wsl --list --quiet 2>$null
    $ErrorActionPreference = 'Stop'
    $listed = ($installed -join "`n").Replace([string][char]0, '')
    if ($listed -notmatch '(?m)^Ubuntu-24\.04\s*$') {
        Write-Host 'Preparing WSL2 + Ubuntu 24.04. Windows may request administrator authorization.'
        Write-Host 'If a reboot is requested, reboot Windows and launch this BAT again.'
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            $setup = Start-Process -FilePath $wsl -Verb RunAs -ArgumentList '--install -d Ubuntu-24.04 --no-launch' -Wait -PassThru
            if ($setup.ExitCode -ne 0) { Stop-Setup ('WSL installation did not finish (exit ' + $setup.ExitCode + '). Reboot if requested and run again.') }
        } else {
            & $wsl --install -d Ubuntu-24.04 --no-launch
            if ($LASTEXITCODE -ne 0) { Stop-Setup 'WSL installation needs attention. Reboot if requested and run again.' }
        }
    }
    & $wsl --set-version $distro 2
    if ($LASTEXITCODE -ne 0) { Stop-Setup 'Could not start/convert Ubuntu to WSL2. Reboot if requested, enable CPU virtualization in BIOS, run wsl --update, then retry.' }
    # Create clouds only if necessary, using the already provided Windows converter.
    $cloudRoot = Join-Path $ProjectRoot 'data\MCB_pointcloud'
    $variants = @('area_random_2048','area_random_4096','area_random_8192','fps_2048','fps_4096','fps_8192','poisson_disk_2048','poisson_disk_4096','poisson_disk_8192')
    $missing = $false
    $reconverted = '0'
    foreach ($variant in $variants) {
        if (-not (Test-Path (Join-Path $cloudRoot "$variant\train")) -or -not (Test-Path (Join-Path $cloudRoot "$variant\test"))) { $missing = $true }
    }
    if ($missing) {
        Write-Host 'Some point-cloud variants are missing. Running the existing 9-set converter...'
        $converter = Join-Path $ProjectRoot 'data\_pointcloud_converter\bootstrap.ps1'
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $converter -DataRoot (Join-Path $ProjectRoot 'data')
        if ($LASTEXITCODE -ne 0) { Stop-Setup 'Point-cloud conversion failed. Fix the converter error and retry.' }
        $reconverted = '1'
    }
    $linuxRoot = (& $wsl --distribution $distro --user root --exec wslpath -a -u $ProjectRoot | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($linuxRoot)) { Stop-Setup 'Could not translate the project path to a WSL path.' }
    $logDir = Join-Path $ProjectRoot 'results\setup'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $logPath = Join-Path $logDir ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    Write-Host ('Running profile: ' + $Profile)
    Write-Host ('Setup and suite log: ' + $logPath)
    # Windows PowerShell 5.1 wraps native stderr as ErrorRecord. CUDA compiler/pip
    # warnings must be logged, rather than aborting a successful native process.
    $ErrorActionPreference = 'Continue'
    & $wsl --distribution $distro --user root --exec bash "$linuxRoot/experiments/bootstrap_wsl.sh" $linuxRoot $Profile $reconverted 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($exitCode -ne 0) { Stop-Setup ('Setup or experiments failed (exit ' + $exitCode + '). See logs. Rerun the same BAT to resume; completed tasks will be retained.') }
    Write-Host ('Finished. Raw records: ' + (Join-Path $ProjectRoot "results\$Profile"))
    exit 0
} catch {
    Write-Host ''
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
