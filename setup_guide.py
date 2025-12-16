"""
Interactive Setup Guide for Insurance Q&A Chatbot
Run this script to check prerequisites and guide through setup
"""

import sys
import os
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")


def print_step(number, text):
    """Print step number and description"""
    print(f"\n{'─' * 70}")
    print(f"STEP {number}: {text}")
    print('─' * 70)


def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False


def check_pip():
    """Check if pip is available"""
    print_step(2, "Checking pip")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        print(result.stdout.strip())
        print("✅ pip is available")
        return True
    except Exception as e:
        print(f"❌ pip not found: {e}")
        return False


def install_requirements():
    """Install required packages"""
    print_step(3, "Installing Requirements")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    print("Installing packages from requirements.txt...")
    print("This may take a few minutes...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ All packages installed successfully")
            return True
        else:
            print(f"❌ Installation failed:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing packages: {e}")
        return False


def check_env_file():
    """Check if .env file exists and guide creation"""
    print_step(4, "Checking Environment Variables")
    
    if Path(".env").exists():
        print("✅ .env file found")
        
        # Read and check for required variables
        with open(".env", "r") as f:
            content = f.read()
            
        required_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
        missing = []
        
        for var in required_vars:
            if var not in content or f"{var}=" in content and "your_" in content:
                missing.append(var)
        
        if missing:
            print(f"⚠️  Please set these variables in .env: {', '.join(missing)}")
            return False
        else:
            print("✅ Required environment variables are set")
            return True
    else:
        print("❌ .env file not found")
        print("\nCreating .env file from template...")
        
        if Path("env_template.txt").exists():
            with open("env_template.txt", "r") as f:
                template = f.read()
            
            with open(".env", "w") as f:
                f.write(template)
            
            print("✅ .env file created")
            print("\n⚠️  IMPORTANT: Edit .env file and add your Pinecone API key!")
            print("   Get your API key from: https://app.pinecone.io/")
            return False
        else:
            print("❌ env_template.txt not found")
            return False


def check_ollama():
    """Check if Ollama is installed and running"""
    print_step(5, "Checking Ollama")
    
    try:
        # Check if ollama command exists
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ Ollama is installed and running")
            
            # Check if mistral model is available
            if "mistral" in result.stdout:
                print("✅ Mistral model is installed")
                return True
            else:
                print("⚠️  Mistral model not found")
                print("\nTo install Mistral, run:")
                print("   ollama pull mistral")
                return False
        else:
            print("❌ Ollama is not running")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        print("\nTo install Ollama:")
        print("   1. Visit https://ollama.ai")
        print("   2. Download and install for your OS")
        print("   3. Run: ollama pull mistral")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def test_imports():
    """Test if key packages can be imported"""
    print_step(6, "Testing Package Imports")
    
    packages = [
        ("streamlit", "Streamlit"),
        ("datasets", "Hugging Face Datasets"),
        ("sentence_transformers", "Sentence Transformers"),
        ("pinecone", "Pinecone"),
        ("dotenv", "Python Dotenv")
    ]
    
    all_imported = True
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            all_imported = False
    
    return all_imported


def run_pinecone_setup():
    """Ask if user wants to run Pinecone setup"""
    print_step(7, "Pinecone Database Setup")
    
    print("The next step is to set up the Pinecone database.")
    print("This will:")
    print("  - Download the InsuranceQA-v2 dataset")
    print("  - Generate embeddings for all Q&A pairs")
    print("  - Store vectors in Pinecone")
    print("  - Take approximately 5-10 minutes")
    print("\n⚠️  Make sure your Pinecone API key is set in .env first!")
    
    response = input("\nDo you want to run the setup now? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nRunning pinecone_setup.py...")
        try:
            subprocess.run([sys.executable, "pinecone_setup.py"])
            return True
        except Exception as e:
            print(f"❌ Error running setup: {e}")
            return False
    else:
        print("\nYou can run it later with:")
        print("   python pinecone_setup.py")
        return False


def main():
    """Main setup guide function"""
    print_header("Insurance Q&A Chatbot - Setup Guide")
    
    print("This script will guide you through setting up the chatbot.")
    print("Press Ctrl+C at any time to exit.")
    
    # Run checks
    checks = [
        check_python_version(),
        check_pip()
    ]
    
    if not all(checks):
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Install requirements
    response = input("\nInstall required packages? (y/n): ").strip().lower()
    if response == 'y':
        install_requirements()
    
    # Check environment
    env_ready = check_env_file()
    
    # Check Ollama
    ollama_ready = check_ollama()
    
    # Test imports
    test_imports()
    
    # Summary
    print_header("Setup Summary")
    
    if env_ready and ollama_ready:
        print("✅ All prerequisites are ready!")
        
        # Offer to run Pinecone setup
        run_pinecone_setup()
        
        print_header("Setup Complete!")
        print("\nYou're all set! To start the chatbot, run:")
        print("   streamlit run app.py")
        print("\nOn first use, click 'Initialize Chatbot' in the sidebar.")
    else:
        print("⚠️  Some prerequisites need attention:")
        
        if not env_ready:
            print("   - Set up .env file with Pinecone credentials")
        
        if not ollama_ready:
            print("   - Install Ollama and pull Mistral model")
        
        print("\nAfter fixing these, run this script again or proceed manually.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)


