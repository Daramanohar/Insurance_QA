"""
Create Submission Package
This script creates a .zip file for assignment submission
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime


def create_submission_zip():
    """
    Create a zip file containing all project files for submission
    """
    print("=" * 70)
    print(" Creating Submission Package")
    print("=" * 70)
    
    # Files to include in submission
    files_to_include = [
        "app.py",
        "config.py",
        "data_loader.py",
        "pinecone_setup.py",
        "ollama_client.py",
        "requirements.txt",
        "env_template.txt",
        "README.md",
        "QUICKSTART.md",
        "PRESENTATION_GUIDE.md",
        "setup_guide.py",
        "test_system.py",
        ".gitignore"
    ]
    
    # Create zip filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"Insurance_QA_Chatbot_Submission_{timestamp}.zip"
    
    print(f"\nCreating: {zip_filename}")
    print("\nIncluding files:")
    
    # Create zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in files_to_include:
            if Path(filename).exists():
                zipf.write(filename)
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (not found)")
    
    # Get file size
    file_size = os.path.getsize(zip_filename)
    size_mb = file_size / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print(f"✅ Submission package created successfully!")
    print(f"   File: {zip_filename}")
    print(f"   Size: {size_mb:.2f} MB")
    print("=" * 70)
    
    print("\n📋 Submission Checklist:")
    print("  [ ] All code files included")
    print("  [ ] README.md with setup instructions")
    print("  [ ] requirements.txt")
    print("  [ ] Environment template (env_template.txt)")
    print("  [ ] Setup and test scripts")
    print("  [ ] Documentation files")
    print("  [ ] PowerPoint presentation (create separately)")
    
    print("\n⚠️  Remember to also submit:")
    print("  - PowerPoint presentation (.pptx)")
    print("  - Demo video (optional but recommended)")
    
    print("\n💡 Tips:")
    print("  - Test the submission by extracting and running in a fresh directory")
    print("  - Ensure .env is NOT included (use env_template.txt instead)")
    print("  - Include screenshots in your presentation")
    
    return zip_filename


def verify_submission():
    """
    Verify that all required files exist before creating submission
    """
    print("\nVerifying project files...")
    
    required_files = [
        "app.py",
        "config.py",
        "data_loader.py",
        "pinecone_setup.py",
        "ollama_client.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    
    for filename in required_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} required file(s) missing!")
        return False
    else:
        print("\n✅ All required files present")
        return True


if __name__ == "__main__":
    print("\n🎓 Insurance Q&A Chatbot - Submission Package Creator\n")
    
    # Verify files
    if verify_submission():
        # Create submission zip
        zip_file = create_submission_zip()
        print(f"\n✨ Done! Submit: {zip_file}")
    else:
        print("\n❌ Cannot create submission package. Fix missing files first.")


