REM Benchmarking batch file for OpenVINO models. Not intended to run as-is, but as a starting point.
REM Run this script in a Python environment where `pip install openvino` is installed. Python 3.9-3.12 are supported
REM On systems with GPU/NPU, most recent drivers should be installed
REM Power mode should be "Performance" and laptops should be plugged in
REM USUAGE: benchmark.bat /path/to/model /path/to/logfile (logfile will be created)

set MODEL=%~1
set LOG=%~2
set LOGTXT="%LOG%.txt"
set LOGCSV="%LOG%.csv"

REM save system info and OpenVINO properties to log
systeminfo > %LOGTXT%
powershell.exe -Command "Get-CimInstance -ClassName Win32_Processor | Format-List * " >> %LOGTXT%
powershell -Command "Get-CimInstance -ClassName Win32_VideoController | Select-Object DriverVersion, DriverDate, Description | Format-List" >> %LOGTXT%
powercfg /GetActiveScheme >> %LOGTXT%

REM curl -O https://raw.githubusercontent.com/helena-intel/snippets/refs/heads/main/show_properties/show_compiled_model_properties.py
REM python show_compiled_model_properties.py %MODEL% >> %LOGTXT%

REM run benchmark three times on all available devices
for /L %%i in (1,1,1) do @python sync_benchmark.py %MODEL% ALL --log %LOGCSV%

REM create openvinologs.zip with .txt and .csv files
powershell.exe -Command "if (Test-Path 'openvinologs.zip') { Remove-Item 'openvinologs.zip' }; Compress-Archive -Path *.csv, *.txt -DestinationPath 'openvinologs.zip'"
