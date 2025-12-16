# Fix PyTorch DLL Error on Windows

## The Problem

Error: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

This happens because PyTorch needs Microsoft Visual C++ Redistributables, which are missing from your system.

---

## Solution: Install Visual C++ Redistributables

### Method 1: Direct Download (Recommended)

1. **Download** the installer:
   - Visit: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Or search: "Microsoft Visual C++ Redistributable latest"

2. **Run** the downloaded file (`vc_redist.x64.exe`)

3. **Click "Install"** and wait for completion

4. **Restart** your computer (important!)

5. **Run** your app again:
   ```bash
   python -m streamlit run app.py
   ```

---

### Method 2: Install via winget (Windows 10+)

```powershell
winget install Microsoft.VCRedist.2015+.x64
```

Then restart your computer.

---

### Method 3: Install All Visual C++ Versions

Download and install from Microsoft:
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

Install both:
- vc_redist.x64.exe (64-bit)
- vc_redist.x86.exe (32-bit) - optional but recommended

---

## After Installation

1. **Restart** your computer (required!)

2. **Test** if PyTorch works:
   ```bash
   python -c "import torch; print('PyTorch works!'); print(f'Version: {torch.__version__}')"
   ```

3. **Run** your chatbot:
   ```bash
   python -m streamlit run app.py
   ```

---

## Alternative: Use Workaround (If Installation Fails)

If you cannot install Visual C++ Redistributables, we can create a workaround that lazy-loads PyTorch only when needed.

Run this to create a workaround version:
```bash
python create_workaround.py
```

---

## Check Your System

See what's currently installed:

```powershell
Get-ItemProperty HKLM:\Software\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall\* | Where-Object { $_.DisplayName -like "*Visual C++*" } | Select-Object DisplayName, DisplayVersion
```

---

## Common Issues

### Issue 1: "Already Installed"
- Uninstall old versions first
- Install the latest version
- Restart

### Issue 2: "Installation Failed"
- Run as Administrator
- Disable antivirus temporarily
- Check Windows Update is current

### Issue 3: "Still Not Working"
- Restart is REQUIRED after installation
- Try installing both x64 and x86 versions
- Check if Windows is up to date

---

## Why This Happens

PyTorch uses C++ libraries that require Microsoft's Visual C++ runtime libraries. Windows doesn't include these by default, so they must be installed separately.

These are the same libraries needed by many Windows applications and games, so installing them is safe and common.

---

## Next Steps

1. ✅ Install Visual C++ Redistributables
2. ✅ Restart your computer
3. ✅ Test: `python -c "import torch; print('Works!')"`
4. ✅ Run: `python -m streamlit run app.py`

Your chatbot will work after this!

