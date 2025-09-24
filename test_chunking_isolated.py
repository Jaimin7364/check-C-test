#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from ci_pr_test_runner import CDependencyAnalyzer, CFunctionInfo, FunctionSignature, CIncludeInfo

def test_isolated_function_detection():
    """Test that add2 function can be detected as isolated"""
    
    # Create a mock project structure
    project_root = Path(__file__).parent
    analyzer = CDependencyAnalyzer(project_root)
    
    # Create mock functions from main.c
    functions = []
    
    # add function - called by main
    add_func = CFunctionInfo(
        name="add",
        file_path="main.c", 
        code="int add(int a, int b, int c) {\n    return a + b + c;\n}",
        line_start=3,
        line_end=5,
        signature=FunctionSignature(name="add", return_type="int", parameters=[
            {"type": "int", "name": "a"},
            {"type": "int", "name": "b"}, 
            {"type": "int", "name": "c"}
        ]),
        include_info=CIncludeInfo(),
        complexity_score=1
    )
    functions.append(add_func)
    
    # add2 function - also called by main, but might be considered isolated if newly added
    add2_func = CFunctionInfo(
        name="add2",
        file_path="main.c",
        code="int add2(int a, int b) {\n    return a + b;\n}",
        line_start=7,
        line_end=9,
        signature=FunctionSignature(name="add2", return_type="int", parameters=[
            {"type": "int", "name": "a"},
            {"type": "int", "name": "b"}
        ]),
        include_info=CIncludeInfo(),
        complexity_score=1
    )
    functions.append(add2_func)
    
    # main function - calls both add and add2
    main_func = CFunctionInfo(
        name="main",
        file_path="main.c",
        code="""int main() {
    int num1, num2, num3, result;
    
    printf("Enter three numbers: ");
    scanf("%d %d %d", &num1, &num2, &num3);
    
    result = add(num1, num2, num3);
    
    printf("Sum: %d\\n", result);
    result = add2(num1, num2);
    printf("Sum2: %d\\n", result);
    return 0;
}""",
        line_start=11,
        line_end=23,
        signature=FunctionSignature(name="main", return_type="int", parameters=[]),
        include_info=CIncludeInfo(),
        complexity_score=2
    )
    functions.append(main_func)
    
    # Build dependency map
    dependency_map = analyzer.build_dependency_map(["main.c"])
    
    print("🔍 Testing isolated function detection...")
    print("=" * 50)
    
    for func in functions:
        can_isolate = analyzer._can_function_be_isolated(func, functions, dependency_map)
        print(f"Function '{func.name}': {'✅ CAN BE ISOLATED' if can_isolate else '❌ NEEDS INTEGRATION TESTING'}")
        
        # Show reasoning
        if func.name == "add2":
            print(f"  Reasoning for {func.name}:")
            call_pattern = analyzer.re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
            calls = set()
            for match in call_pattern.finditer(func.code):
                call_name = match.group(1)
                if call_name not in {'if', 'while', 'for', 'switch', 'sizeof', 'return', 
                                   'printf', 'scanf', 'malloc', 'free'}:
                    calls.add(call_name)
            
            user_functions = {f.name for f in functions}
            user_calls = calls.intersection(user_functions)
            user_calls.discard(func.name)
            
            print(f"    - Calls user functions: {user_calls}")
            print(f"    - Called by other functions: {func.name in ['add', 'add2'] and 'main' in [f.name for f in functions]}")
    
    print("=" * 50)
    
    # Test chunk creation with isolated detection
    print("🧩 Testing chunk creation...")
    
    # Simulate changes to add2 function only
    from ci_pr_test_runner import CodeChange
    changes = [
        CodeChange(
            file_path="main.c",
            line_start=7,
            line_end=9,
            change_type="modified",
            content="int add2(int a, int b) {\n    return a + b;\n}"
        )
    ]
    
    chunks = analyzer.create_test_chunks(changes, functions)
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.test_type} testing ({len(chunk.dependent_functions)} functions)")
        for func in chunk.dependent_functions:
            print(f"    - {func.name}")

if __name__ == "__main__":
    test_isolated_function_detection()