#!/usr/bin/env python3
"""
Mass deletion script for unused code in voice-to-voice application.
This script removes files that are confirmed to be unused and safe to delete.
"""

import os
import sys
from pathlib import Path
from typing import List

def get_files_to_delete() -> List[Path]:
    """Return list of files that can be safely deleted"""
    
    # Files in root directory that are standalone test files
    root_test_files = [
        "test_enhanced_summary.py",
        "test_summary_prompts.py", 
        "test_vad_improvements.py",
        "test_interruption_logging.py"
    ]
    
    # Broken integration test file
    broken_test_files = [
        "local-voice-to-voice/tests/integration/test_pipeline.py"
    ]
    
    # Additional potentially redundant files (user should review)
    potentially_redundant_files = [
        # Uncomment these after reviewing if they're still needed:
        # "local-voice-to-voice/fast_chatbot.py",  # Duplicates main.py functionality
        # "local-voice-to-voice/quick_test.py",    # Simple test, could be in examples
    ]
    
    files_to_delete = []
    
    # Add root test files
    for filename in root_test_files:
        file_path = Path(filename)
        if file_path.exists():
            files_to_delete.append(file_path)
    
    # Add broken integration test
    for filename in broken_test_files:
        file_path = Path(filename)
        if file_path.exists():
            files_to_delete.append(file_path)
    
    return files_to_delete

def get_code_quality_issues() -> dict:
    """Return dictionary of code quality issues found"""
    return {
        "duplicate_methods": [
            {
                "method": "_is_similar_text",
                "locations": [
                    "src/pipeline/voice_assistant.py:984-1014",
                    "src/pipeline/conversation_logger.py:72-102"
                ],
                "recommendation": "Extract to shared utility module"
            }
        ],
        "god_classes": [
            {
                "class": "VoiceAssistant",
                "file": "src/pipeline/voice_assistant.py",
                "lines": 1373,
                "responsibilities": ["STT", "TTS", "LLM", "Audio", "Logging", "Memory"],
                "recommendation": "Break into specialized components"
            }
        ],
        "magic_numbers": [
            "similarity_threshold=0.8 (appears in multiple files)",
            "target_duration=12.0 (trim_voice_sample.py)",
            "Various hardcoded timeouts"
        ],
        "excessive_parameters": [
            {
                "method": "VoiceAssistant.__init__",
                "parameter_count": "25+",
                "recommendation": "Use configuration objects"
            }
        ]
    }

def print_banner():
    """Print cleanup banner"""
    print("🧹" * 60)
    print("🗑️  VOICE-TO-VOICE CODE CLEANUP SCRIPT")
    print("🧹" * 60)
    print()

def print_file_analysis(files_to_delete: List[Path]):
    """Print detailed analysis of files to be deleted"""
    print("📋 FILES IDENTIFIED FOR DELETION:")
    print("=" * 50)
    
    if not files_to_delete:
        print("✅ No unused files found for automatic deletion!")
    else:
        # Categorize files
        root_tests = []
        integration_tests = []
        
        for file_path in files_to_delete:
            if str(file_path).startswith("test_"):
                root_tests.append(file_path)
            elif "integration" in str(file_path):
                integration_tests.append(file_path)
        
        if root_tests:
            print("\n🧪 STANDALONE TEST FILES (Root Directory):")
            print("   These are development/debugging scripts not integrated into main app")
            for file_path in root_tests:
                size = file_path.stat().st_size if file_path.exists() else 0
                print(f"   ❌ {file_path} ({size} bytes)")
        
        if integration_tests:
            print("\n💥 BROKEN INTEGRATION TESTS:")
            print("   These import non-existent classes and cannot run")
            for file_path in integration_tests:
                size = file_path.stat().st_size if file_path.exists() else 0
                print(f"   ❌ {file_path} ({size} bytes)")
        
        total_size = sum(f.stat().st_size for f in files_to_delete if f.exists())
        print(f"\n📊 TOTAL: {len(files_to_delete)} files, {total_size:,} bytes")

def print_code_quality_analysis():
    """Print code quality issues found"""
    print("\n🔍 CODE QUALITY ANALYSIS:")
    print("=" * 50)
    
    issues = get_code_quality_issues()
    
    if issues["duplicate_methods"]:
        print("\n🔄 DUPLICATE CODE FOUND:")
        for dup in issues["duplicate_methods"]:
            print(f"   📝 Method: {dup['method']}")
            for location in dup["locations"]:
                print(f"      📍 {location}")
            print(f"      💡 {dup['recommendation']}")
    
    if issues["god_classes"]:
        print("\n🏗️  GOD CLASSES (TOO LARGE):")
        for god_class in issues["god_classes"]:
            print(f"   📦 Class: {god_class['class']} ({god_class['lines']} lines)")
            print(f"      📍 {god_class['file']}")
            print(f"      🎯 Handles: {', '.join(god_class['responsibilities'])}")
            print(f"      💡 {god_class['recommendation']}")
    
    if issues["excessive_parameters"]:
        print("\n⚙️  EXCESSIVE PARAMETERS:")
        for param_issue in issues["excessive_parameters"]:
            print(f"   🔧 {param_issue['method']} ({param_issue['parameter_count']} parameters)")
            print(f"      💡 {param_issue['recommendation']}")
    
    if issues["magic_numbers"]:
        print("\n🪄 MAGIC NUMBERS:")
        for magic in issues["magic_numbers"]:
            print(f"   🔢 {magic}")
    
    print("\n💡 REFACTORING RECOMMENDATIONS:")
    print("   1. Create src/utils/text_similarity.py for shared text comparison")
    print("   2. Create src/utils/text_cleaner.py for text preprocessing")
    print("   3. Break VoiceAssistant into smaller, focused classes")
    print("   4. Use configuration objects instead of long parameter lists")
    print("   5. Extract hardcoded values to configuration constants")

def confirm_deletion() -> bool:
    """Ask user for confirmation"""
    print("\n" + "⚠️ " * 20)
    print("⚠️  CONFIRMATION REQUIRED")
    print("⚠️ " * 20)
    print("\nThese files will be PERMANENTLY DELETED.")
    print("Make sure you have a backup if needed!")
    print("\nAre you sure you want to proceed? (yes/no): ", end="")
    
    response = input().strip().lower()
    return response in ['yes', 'y']

def delete_files(files_to_delete: List[Path]) -> tuple[int, int]:
    """Delete the files and return (success_count, error_count)"""
    success_count = 0
    error_count = 0
    
    print("\n🗑️  DELETING FILES...")
    print("-" * 30)
    
    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"✅ Deleted: {file_path}")
                success_count += 1
            else:
                print(f"⚠️  Not found: {file_path}")
        except Exception as e:
            print(f"❌ Error deleting {file_path}: {e}")
            error_count += 1
    
    return success_count, error_count

def cleanup_empty_directories():
    """Remove empty directories after file deletion"""
    print("\n📁 CLEANING UP EMPTY DIRECTORIES...")
    
    # Check if integration test directory is empty
    integration_dir = Path("local-voice-to-voice/tests/integration")
    if integration_dir.exists() and not any(integration_dir.iterdir()):
        try:
            integration_dir.rmdir()
            print(f"✅ Removed empty directory: {integration_dir}")
        except Exception as e:
            print(f"⚠️  Could not remove {integration_dir}: {e}")

def print_summary(success_count: int, error_count: int):
    """Print cleanup summary"""
    print("\n" + "🎉" * 60)
    print("📊 CLEANUP SUMMARY")
    print("🎉" * 60)
    
    if success_count > 0:
        print(f"✅ Successfully deleted: {success_count} files")
    
    if error_count > 0:
        print(f"❌ Errors encountered: {error_count} files")
        print("   Check error messages above for details")
    
    if success_count > 0 and error_count == 0:
        print("\n🎯 CLEANUP COMPLETE!")
        print("Your voice-to-voice application is now leaner.")
        print("\nNext steps for code quality improvement:")
        print("1. Refactor duplicate _is_similar_text methods")
        print("2. Consider breaking down VoiceAssistant class (1373 lines)")
        print("3. Extract hardcoded values to configuration")
        print("4. Review potentially redundant files manually:")
        print("   - fast_chatbot.py (alternative implementation)")
        print("   - analyze_performance.py (performance analysis)")
        print("   - quick_test.py (simple VAD test)")
    
    print("\n🧹 Happy coding!")

def main():
    """Main cleanup function"""
    print_banner()
    
    # Get files to delete
    files_to_delete = get_files_to_delete()
    
    # Show file analysis
    print_file_analysis(files_to_delete)
    
    # Show code quality analysis
    print_code_quality_analysis()
    
    if not files_to_delete:
        print("\n✨ No files to delete, but code quality improvements recommended above.")
        return 0
    
    # Get confirmation
    if not confirm_deletion():
        print("\n🚫 Cleanup cancelled by user.")
        print("No files were deleted.")
        return 0
    
    # Delete files
    success_count, error_count = delete_files(files_to_delete)
    
    # Cleanup empty directories
    cleanup_empty_directories()
    
    # Print summary
    print_summary(success_count, error_count)
    
    return 1 if error_count > 0 else 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🚫 Cleanup interrupted by user (Ctrl+C)")
        print("No files were deleted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 