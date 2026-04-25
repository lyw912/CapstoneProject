"""
Simple test runner script

Run this script directly to execute tests
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test_monitor import TestLogMonitor


def main():
    """Run all tests"""
    print("=" * 60)
    print("ForumEngine Log Parsing Tests")
    print("=" * 60)
    print()
    
    test_instance = TestLogMonitor()
    test_instance.setup_method()
    
    # Get all test methods
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for test_method_name in test_methods:
        test_method = getattr(test_instance, test_method_name)
        print(f"Running test: {test_method_name}...", end=" ")
        
        try:
            test_method()
            print("✓ Passed")
            passed += 1
        except AssertionError as e:
            print(f"✗ Failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Test Results: {passed} Passed, {failed} Failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

