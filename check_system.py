"""
Check system for required dependencies
"""

import sys
import subprocess

print("=" * 70)
print(" System Dependency Check")
print("=" * 70)

print("\n1. Python Version:")
print(f"   {sys.version}")

print("\n2. Checking PyTorch:")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__} - OK")
except Exception as e:
    print(f"   ✗ PyTorch - FAILED")
    print(f"   Error: {str(e)[:100]}")
    print("\n   >>> FIX: Install Visual C++ Redistributables")
    print("   >>> Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")

print("\n3. Checking other dependencies:")

packages = [
    'streamlit',
    'sentence_transformers',
    'pinecone',
    'datasets',
    'pandas'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg} - OK")
    except ImportError:
        print(f"   ✗ {pkg} - Missing")
    except Exception as e:
        print(f"   ⚠ {pkg} - Error: {str(e)[:50]}")

print("\n4. Checking Visual C++ (Windows):")
if sys.platform == 'win32':
    try:
        result = subprocess.run(
            ['powershell', '-Command', 
             "Get-ItemProperty HKLM:\\Software\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* | Where-Object { $_.DisplayName -like '*Visual C++*' } | Select-Object DisplayName -First 5"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.stdout and 'Visual C++' in result.stdout:
            print("   ✓ Visual C++ Redistributables found")
            # Show first few lines
            lines = [l for l in result.stdout.split('\n') if 'Visual C++' in l]
            for line in lines[:3]:
                print(f"     {line.strip()}")
        else:
            print("   ⚠ Visual C++ Redistributables NOT found")
            print("   >>> This is likely causing the PyTorch error!")
            print("   >>> Install from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    except Exception as e:
        print(f"   ? Could not check: {e}")
else:
    print("   (Not on Windows)")

print("\n" + "=" * 70)
print(" Summary")
print("=" * 70)

print("\nIf PyTorch shows an error above:")
print("  1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
print("  2. Install the downloaded file")
print("  3. RESTART your computer")
print("  4. Run this check again: python check_system.py")

print("\n" + "=" * 70)

