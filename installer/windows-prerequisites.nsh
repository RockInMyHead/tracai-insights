!macro customInstall
  DetailPrint "Checking TrackAI Windows prerequisites..."
  IfFileExists "$INSTDIR\resources\local-gpu\prerequisites\vc_redist.x64.exe" 0 +4
    DetailPrint "Installing Microsoft Visual C++ Redistributable x64..."
    ExecWait '"$INSTDIR\resources\local-gpu\prerequisites\vc_redist.x64.exe" /install /quiet /norestart' $0
    DetailPrint "Microsoft Visual C++ Redistributable installer exit code: $0"
  ClearErrors
  nsExec::ExecToStack 'where nvidia-smi'
  Pop $0
  Pop $1
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "TrackAI GPU runtime includes CUDA/PyTorch, but Windows does not report an NVIDIA display driver. Install the NVIDIA driver for this PC GPU, then restart TrackAI."
  ${Else}
    DetailPrint "NVIDIA driver tool detected: $1"
  ${EndIf}
!macroend
