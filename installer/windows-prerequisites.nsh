!macro customInstall
  DetailPrint "Checking TrackAI Windows prerequisites..."
  IfFileExists "$INSTDIR\resources\local-gpu\prerequisites\vc_redist.x64.exe" 0 +4
    DetailPrint "Installing Microsoft Visual C++ Redistributable x64..."
    ExecWait '"$INSTDIR\resources\local-gpu\prerequisites\vc_redist.x64.exe" /install /quiet /norestart' $0
    DetailPrint "Microsoft Visual C++ Redistributable installer exit code: $0"
!macroend
