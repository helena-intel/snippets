# PowerShell script to check NPU driver version on Windows AI PC with Intel NPU. The script checks the currently
# installed (and in-use) NPU driver against a hardcoded latest version and shows if the driver is up-to-date
#
# Run from either cmd.exe or PowerShell with (-NoProfile and -ExecutionPolicy Bypass are optional but included for broadest compatibility):
# powershell.exe -NoProfile -ExecutionPolicy Bypass -File check_npu_driver.ps1
# To run this script without manually downloading it, use:
# powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "iex (irm 'https://raw.githubusercontent.com/helena-intel/snippets/refs/heads/main/utils/check_npu_driver.ps1')"

$deviceNameMatch = "AI Boost"
$latestDriverVersion = "32.0.100.4023"
$driverUpdateURL = "https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html"

# Get current driver version
$currentDriverVersion = (Get-PnpDevice |
    Where-Object { $_.FriendlyName -match $deviceNameMatch -and $_.Status -eq "OK" } |
    ForEach-Object {
        (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName "DEVPKEY_Device_DriverVersion").Data
    })

if (-not $currentDriverVersion) {
    Write-Output "No NPU device ('$deviceNameMatch') or driver found."
    Write-Output "If you have an NPU, go to $driverUpdateURL to install the driver."
    exit 1
}

Write-Output "Your NPU driver version is $currentDriverVersion."

# Check if currently active driver is latest
if ($currentDriverVersion -eq $latestDriverVersion) {
    Write-Output "The driver is up-to-date."
}
else {
    # Check if latest driver is installed (but inactive)
    # Not 100% robust, but unlikely that there is another device with exactly the same driver version as the NPU driver
    $pnputilOutput = pnputil /enum-drivers | Select-String -SimpleMatch $latestDriverVersion

    if ($pnputilOutput) {
        Write-Output "You have the latest driver (version $latestDriverVersion) installed, but it is not currently active."
        Write-Output "See https://github.com/helena-intel/readmes/blob/main/npu_driver_revert.md"
    }
    else {
        # Latest driver not installed
        Write-Output "The latest driver version is $latestDriverVersion."
        Write-Output "Go to $driverUpdateURL to update the driver."
    }
}
