# Summon installer for Windows - AI, Open-source, Private Intelligence.
#
#   irm https://codyvolckmar.github.io/summon/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://codyvolckmar.github.io/summon/install.ps1))) --dry-run
#   & ([scriptblock]::Create((irm https://codyvolckmar.github.io/summon/install.ps1))) ornith9b 40000
#
# Runs Summon inside WSL 2 (Ubuntu), which has direct access to your NVIDIA GPU. The AI endpoint is then
# http://localhost:PORT/v1 on this PC (local mode: nothing else on your network can reach it).
# Needs: Windows 10 21H2+ or 11, an NVIDIA GPU with a current Windows driver, WSL 2.
# Read before you run: https://github.com/codyvolckmar/summon/blob/main/docs/install.ps1

$ErrorActionPreference = 'Stop'
$Base   = if ($env:SUMMON_BASE)   { $env:SUMMON_BASE }   else { 'https://codyvolckmar.github.io/summon' }
$Distro = if ($env:SUMMON_DISTRO) { $env:SUMMON_DISTRO } else { 'Ubuntu' }
$PassArgs = @($args)
if ($PassArgs.Count -eq 0 -and $env:SUMMON_ARGS) { $PassArgs = $env:SUMMON_ARGS -split '\s+' }

function Say($text, $color = 'Gray') { Write-Host $text -ForegroundColor $color }

Say ''
Say '    ____    _   _   __  __   __  __    ___    _   _' Magenta
Say '   / ___|  | | | | |  \/  | |  \/  |  / _ \  | \ | |' Magenta
Say '   \___ \  | | | | | |\/| | | |\/| | | | | | |  \| |' Magenta
Say '    ___) | | |_| | | |  | | | |  | | | |_| | | |\  |' Magenta
Say '   |____/   \___/  |_|  |_| |_|  |_|  \___/  |_| \_|' Magenta
Say '         *  A I  .  Open-source  .  Private  .  Intelligence  *' Cyan
Say '   Windows edition: Summon runs inside WSL 2 with your NVIDIA GPU.' Cyan
Say ''

# 1. NVIDIA driver on Windows (WSL uses the Windows driver; never install a Linux driver inside WSL)
$dryRun = $PassArgs -contains '--dry-run'
if (-not $dryRun -and -not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
  Say 'No NVIDIA driver found (nvidia-smi.exe). Install the current driver from https://www.nvidia.com/drivers, then run this again.' Yellow
  return
}

# 2. WSL 2 + Ubuntu
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  Say 'WSL is not available on this Windows. Update Windows (10 21H2+ or 11), then run this again.' Yellow
  return
}
$distros = @((wsl.exe -l -q 2>$null) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ })
if ($distros -notcontains $Distro) {
  Say "Installing WSL 2 + $Distro. Windows asks for admin rights; a reboot may follow." Yellow
  Start-Process wsl.exe -ArgumentList "--install -d $Distro" -Verb RunAs -Wait
  Say "When $Distro opens, create your Linux user. Then run this command again to summon your AI." Green
  return
}

# 3. GPU visible inside WSL?
if (-not $dryRun) {
  wsl.exe -d $Distro -- nvidia-smi -L *> $null
  if ($LASTEXITCODE -ne 0) {
    Say "Your GPU isn't visible inside WSL ($Distro). Update the NVIDIA Windows driver and run 'wsl --update', then try again." Yellow
    return
  }
}

# 4. Summon, inside WSL as root (apt, Docker and the GPU runtime need it). Your terminal drives the menus.
$quoted = ($PassArgs | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join ' '
Say "Starting Summon in WSL ($Distro)..." Green
wsl.exe -d $Distro -u root -- bash -c "command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl); curl -fsSL '$Base/install.sh' | bash -s -- $quoted"
if ($LASTEXITCODE -eq 0 -and -not $dryRun) {
  Say ''
  Say 'Your AI endpoint is on this PC: http://localhost:<port>/v1  (the port you chose above).' Green
  Say "Manage it later: wsl -d $Distro -u root -- bash -lc '~/.summon/summon.sh --help'" Gray
}
