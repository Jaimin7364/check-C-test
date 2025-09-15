import os
import subprocess
import json
import sys
import re
import hashlib
import xml.etree.ElementTree as ET
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

from groq import Groq

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY environment variable not set.")
    sys.exit(1)

MODEL_ID = "llama-3.3-70b-versatile"
DEFAULT_TEST_DIR = "tests_pr"

@dataclass
class FunctionSignature:
    """C function signature information"""
    name: str
    return_type: str
    parameters: List[Dict[str, str]] = field(default_factory=list)
    is_static: bool = False
    is_extern: bool = False
    is_inline: bool = False

@dataclass
class CIncludeInfo:
    """Information about includes needed for C functions"""
    system_includes: Set[str] = field(default_factory=set)
    local_includes: Set[str] = field(default_factory=set)
    custom_headers: Set[str] = field(default_factory=set)

@dataclass
class CFunctionInfo:
    """Information about a C function"""
    name: str
    file_path: str
    code: str
    line_start: int
    line_end: int
    signature: FunctionSignature = field(default_factory=lambda: FunctionSignature("", ""))
    include_info: CIncludeInfo = field(default_factory=CIncludeInfo)
    dependencies: Set[str] = field(default_factory=set)
    complexity_score: int = 1
    is_declaration: bool = False

@dataclass
class TestCaseResult:
    """Individual test case result"""
    name: str
    status: str  # "PASS", "FAIL", "ERROR", "SKIP"
    execution_time: float = 0.0
    error_message: str = ""
    failure_reason: str = ""
    test_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "status": self.status,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "failure_reason": self.failure_reason,
            "test_method": self.test_method
        }

@dataclass
class TestReport:
    """Data structure for test execution report"""
    timestamp: str
    model_name: str
    changed_files: List[str]
    analyzed_functions: List[CFunctionInfo]
    test_results: Dict[str, Any]
    coverage_metrics: Dict[str, str]
    status: str
    execution_time: float
    logs: List[str] = field(default_factory=list)
    compilation_status: Dict[str, bool] = field(default_factory=dict)
    test_cases: List[TestCaseResult] = field(default_factory=list)

class CCodeAnalyzer:
    """C code analyzer for function extraction"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.c_keywords = {'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
                          'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
                          'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
                          'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'}
    
    def extract_includes_from_file(self, file_path: Path) -> CIncludeInfo:
        """Extract all includes from a C file"""
        include_info = CIncludeInfo()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find system includes (#include <...>)
            system_includes = re.findall(r'#include\s*<([^>]+)>', content)
            include_info.system_includes.update(system_includes)
            
            # Find local includes (#include "...")
            local_includes = re.findall(r'#include\s*"([^"]+)"', content)
            include_info.local_includes.update(local_includes)
            
            # Identify custom headers (local .h files)
            for inc in local_includes:
                if inc.endswith('.h'):
                    include_info.custom_headers.add(inc)
        
        except FileNotFoundError:
            pass
        
        return include_info
    
    def parse_function_signature(self, func_match: re.Match) -> FunctionSignature:
        """Parse C function signature from regex match"""
        full_signature = func_match.group(0)
        
        # Extract modifiers
        is_static = 'static' in full_signature
        is_extern = 'extern' in full_signature
        is_inline = 'inline' in full_signature
        
        # Extract return type and function name
        signature_line = func_match.group(1).strip()
        
        # Find function name (last identifier before opening parenthesis)
        name_match = re.search(r'(\w+)\s*\(', signature_line)
        if not name_match:
            return FunctionSignature("unknown", "void")
        
        func_name = name_match.group(1)
        
        # Extract return type (everything before function name)
        return_type_part = signature_line[:name_match.start()].strip()
        return_type = self._clean_return_type(return_type_part)
        
        # Extract parameters
        params_match = re.search(r'\((.*?)\)', signature_line, re.DOTALL)
        parameters = []
        if params_match:
            params_str = params_match.group(1).strip()
            if params_str and params_str != 'void':
                parameters = self._parse_parameters(params_str)
        
        return FunctionSignature(
            name=func_name,
            return_type=return_type,
            parameters=parameters,
            is_static=is_static,
            is_extern=is_extern,
            is_inline=is_inline
        )
    
    def _clean_return_type(self, return_type_str: str) -> str:
        """Clean and normalize return type string"""
        # Remove storage specifiers and function specifiers
        modifiers = ['static', 'extern', 'inline', 'register', 'auto']
        for modifier in modifiers:
            return_type_str = return_type_str.replace(modifier, '')
        
        return return_type_str.strip() or 'void'
    
    def _parse_parameters(self, params_str: str) -> List[Dict[str, str]]:
        """Parse function parameters"""
        parameters = []
        if not params_str.strip():
            return parameters
        
        # Split by comma, but be careful with function pointers
        param_parts = self._smart_split_parameters(params_str)
        
        for param in param_parts:
            param = param.strip()
            if param:
                # Simple parameter parsing (can be enhanced)
                parts = param.split()
                if len(parts) >= 2:
                    param_type = ' '.join(parts[:-1])
                    param_name = parts[-1].replace('*', '').replace('[', '').replace(']', '')
                    parameters.append({"type": param_type, "name": param_name})
                elif len(parts) == 1:
                    parameters.append({"type": param, "name": "param"})
        
        return parameters
    
    def _smart_split_parameters(self, params_str: str) -> List[str]:
        """Split parameters by comma, handling nested parentheses"""
        params = []
        current_param = ""
        paren_count = 0
        
        for char in params_str:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                params.append(current_param.strip())
                current_param = ""
                continue
            
            current_param += char
        
        if current_param.strip():
            params.append(current_param.strip())
        
        return params
    
    def calculate_complexity_score(self, code: str) -> int:
        """Calculate complexity score for C function"""
        score = 1
        
        # Count control structures
        control_keywords = ['if', 'else', 'for', 'while', 'do', 'switch', 'case']
        for keyword in control_keywords:
            score += len(re.findall(r'\b' + keyword + r'\b', code))
        
        # Count logical operators
        score += len(re.findall(r'&&|\|\||!', code))
        
        # Count nested blocks (approximate)
        score += code.count('{') // 2
        
        return min(score, 10)  # Cap at 10

class CTestValidator:
    """Validates generated C test code"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def validate_compilation(self, test_code: str, includes: List[str]) -> Tuple[bool, List[str]]:
        """Validate C test code compilation"""
        errors = []
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            
            # Try to compile
            compile_cmd = ['gcc', '-c', '-Wall', '-Wextra'] + [f'-I{inc}' for inc in includes] + [temp_file]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            # Clean up
            os.unlink(temp_file)
            if os.path.exists(temp_file.replace('.c', '.o')):
                os.unlink(temp_file.replace('.c', '.o'))
            
            if result.returncode != 0:
                errors.append(f"Compilation failed: {result.stderr}")
                return False, errors
            
            return True, []
            
        except Exception as e:
            errors.append(f"Compilation validation error: {str(e)}")
            return False, errors

class CChangeAnalyzerAndTester:
    """
    C project analyzer and tester with AI-generated test suite
    """
    def __init__(self):
        self.project_root = Path(os.getcwd())
        
        self.test_dir = self.project_root / DEFAULT_TEST_DIR
        self.test_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.code_analyzer = CCodeAnalyzer(self.project_root)
        self.test_validator = CTestValidator(self.project_root)
        
        self.start_time = datetime.now()
        self.logs = []
        self.model_name = MODEL_ID

    def _log(self, message: str, level: str = "INFO"):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        print(message)

    def _get_changed_files(self) -> List[str]:
        """Gets changed files from the GITHUB_ENV variable."""
        changed_files_str = os.environ.get("CHANGED_FILES", "")
        if not changed_files_str.strip():
            self._log("ℹ️ No changed files detected in this PR.", "INFO")
            return []
        
        changed_files = changed_files_str.split()
        self._log(f"📁 Detected {len(changed_files)} changed files: {', '.join(changed_files)}", "INFO")
        return changed_files

    def _extract_functions_from_c_file(self, file_path: Path) -> List[CFunctionInfo]:
        """Extract functions from C source file"""
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file-level includes
            file_includes = self.code_analyzer.extract_includes_from_file(file_path)
            
            # Pattern to match C function definitions
            # Matches: [modifiers] return_type function_name(params) {
            function_pattern = re.compile(
                r'(?:^|\n)\s*(?:(static|extern|inline)\s+)?'  # Optional modifiers
                r'([a-zA-Z_][a-zA-Z0-9_]*(?:\s*\*\s*)*)\s+'   # Return type
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*'                # Function name  
                r'\([^)]*\)\s*\{',                            # Parameters and opening brace
                re.MULTILINE
            )
            
            # Find all function definitions
            lines = content.splitlines()
            for match in function_pattern.finditer(content):
                try:
                    # Get function signature info
                    signature = self.code_analyzer.parse_function_signature(match)
                    
                    # Find the complete function code
                    start_pos = match.start()
                    start_line = content[:start_pos].count('\n') + 1
                    
                    # Find matching closing brace
                    brace_count = 0
                    func_start = match.end() - 1  # Position of opening brace
                    func_end = func_start
                    
                    for i, char in enumerate(content[func_start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                func_end = func_start + i + 1
                                break
                    
                    end_line = content[:func_end].count('\n') + 1
                    
                    # Extract function code
                    func_lines = lines[start_line-1:end_line]
                    func_code = '\n'.join(func_lines)
                    
                    # Calculate complexity
                    complexity = self.code_analyzer.calculate_complexity_score(func_code)
                    
                    # Create function info
                    func_info = CFunctionInfo(
                        name=signature.name,
                        file_path=str(file_path.relative_to(self.project_root)),
                        code=func_code,
                        line_start=start_line,
                        line_end=end_line,
                        signature=signature,
                        include_info=file_includes,
                        complexity_score=complexity,
                        is_declaration=False
                    )
                    
                    functions.append(func_info)
                    
                except Exception as e:
                    self._log(f"⚠️ Error parsing function at match {match.start()}: {e}", "WARNING")
                    continue
        
        except Exception as e:
            self._log(f"⚠️ Error analyzing {file_path}: {e}", "WARNING")
        
        return functions

    def _build_c_test_prompt(self, functions: List[CFunctionInfo]) -> str:
        """Build comprehensive prompt for C test generation"""
        
        # Analyze all functions to determine includes
        all_system_includes = set()
        all_local_includes = set()
        
        for func in functions:
            all_system_includes.update(func.include_info.system_includes)
            all_local_includes.update(func.include_info.local_includes)
        
        # Build include statements
        include_statements = []
        
        # Standard test includes
        include_statements.extend([
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            "#include <assert.h>",
            "#include <stdarg.h>",
            "#include <setjmp.h>",
            "#include <cmocka.h>  // Unity test framework"
        ])
        
        # Add project-specific includes
        for inc in sorted(all_system_includes):
            include_statements.append(f"#include <{inc}>")
        
        for inc in sorted(all_local_includes):
            include_statements.append(f'#include "{inc}"')
        
        # Create function analysis section
        function_analyses = []
        for i, func in enumerate(functions, 1):
            analysis = [
                f"## Function {i}: {func.signature.name}",
                f"**File**: {func.file_path}",
                f"**Lines**: {func.line_start}-{func.line_end}",
                f"**Return Type**: {func.signature.return_type}",
                f"**Complexity Score**: {func.complexity_score}/10"
            ]
            
            if func.signature.parameters:
                param_info = []
                for param in func.signature.parameters:
                    param_info.append(f"{param['type']} {param['name']}")
                analysis.append(f"**Parameters**: {', '.join(param_info)}")
            else:
                analysis.append("**Parameters**: void")
            
            if func.signature.is_static:
                analysis.append("**Storage**: static")
            if func.signature.is_extern:
                analysis.append("**Storage**: extern") 
            if func.signature.is_inline:
                analysis.append("**Storage**: inline")
            
            analysis.append("**Code**:")
            analysis.append("```c")
            analysis.append(func.code)
            analysis.append("```")
            
            function_analyses.append("\n".join(analysis))
        
        # Build the comprehensive prompt
        prompt = f"""You are an expert C unit testing engineer. Generate comprehensive, compilable C unit tests using the CMocka testing framework.

## CRITICAL REQUIREMENTS FOR C TEST GENERATION:

### 1. COMPILATION AND STRUCTURE:
- Generate ONLY valid C code that compiles without errors
- Use proper C syntax and formatting
- Include all necessary headers
- Use proper function declarations and definitions
- Ensure all braces, parentheses, and semicolons are balanced

### 2. INCLUDES (Use these includes):
```c
{chr(10).join(include_statements)}
```

### 3. TEST STRUCTURE (Use CMocka framework):
```c
// Test function example
static void test_function_name(void **state) {{
    // Setup
    
    // Test execution
    
    // Assertions
    assert_int_equal(expected, actual);
    assert_non_null(pointer);
    // etc.
}}

// Test suite setup
int main(void) {{
    const struct CMUnitTest tests[] = {{
        cmocka_unit_test(test_function_name),
        // Add more tests here
    }};
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}}
```

### 4. FUNCTIONS TO TEST:

{chr(10).join(function_analyses)}

## GENERATION RULES:

### A. For Each Function:
- Create multiple test cases covering different scenarios
- Test normal operation with valid inputs
- Test edge cases and boundary conditions
- Test error conditions and invalid inputs
- Use appropriate CMocka assertions

### B. Memory Management:
- Test functions that allocate/deallocate memory
- Check for memory leaks in tests
- Use proper cleanup in test teardown

### C. Pointer Testing:
- Test NULL pointer handling
- Test pointer arithmetic if applicable
- Validate pointer return values

### D. CMocka Assertions to Use:
- assert_int_equal(a, b) - for integer comparisons
- assert_string_equal(a, b) - for string comparisons  
- assert_non_null(ptr) - for pointer validation
- assert_null(ptr) - for NULL pointer checks
- assert_true(condition) - for boolean conditions
- assert_false(condition) - for negative conditions
- assert_in_range(value, min, max) - for range checks

### E. Mock Strategy (if needed):
- Use CMocka's mock functions for external dependencies
- Mock file I/O operations
- Mock system calls when necessary

## OUTPUT FORMAT:
Generate ONLY a complete C test file with:
1. All necessary #include statements at the top
2. Any required #define macros
3. Static test functions for each function under test
4. Main function with CMocka test suite setup
5. Proper C formatting and syntax

## COMPILATION REQUIREMENTS:
- Code must compile with: gcc -std=c99 -Wall -Wextra
- All functions must be properly declared
- All variables must be declared before use
- Use proper C99 standard compliance

Generate the complete C test file now:"""

        return prompt

    def _invoke_llm_for_c_generation(self, prompt: str, max_retries: int = 3) -> str:
        """Generate C test code using LLM with validation"""
        
        for attempt in range(max_retries):
            try:
                self._log(f"🤖 Generating C test code (attempt {attempt + 1}/{max_retries})...", "INFO")
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a C programming and testing expert. Generate only syntactically correct, compilable C unit test code using CMocka framework. Never include explanations or markdown formatting - only pure C code."},
                        {"role": "user", "content": prompt}
                    ],
                    model=MODEL_ID,
                    temperature=0.1,
                    max_tokens=4000,
                )
                
                generated_code = chat_completion.choices[0].message.content.strip()
                
                # Clean up markdown code fences if present
                if "```c" in generated_code:
                    code_start = generated_code.find("```c") + len("```c")
                    code_end = generated_code.rfind("```")
                    if code_end > code_start:
                        generated_code = generated_code[code_start:code_end].strip()
                elif "```" in generated_code:
                    code_start = generated_code.find("```") + 3
                    code_end = generated_code.rfind("```")
                    if code_end > code_start:
                        generated_code = generated_code[code_start:code_end].strip()
                
                # Basic validation - check if it looks like C code
                if self._basic_c_validation(generated_code):
                    self._log("✅ Generated valid-looking C test code", "SUCCESS")
                    return generated_code
                else:
                    self._log(f"⚠️ Generated code validation failed (attempt {attempt + 1})", "WARNING")
                    if attempt < max_retries - 1:
                        prompt += f"\n\nPREVIOUS ATTEMPT HAD VALIDATION ISSUES. Please ensure:\n- All functions are properly declared\n- All includes are present\n- Proper C syntax is used\n- CMocka assertions are used correctly"
                    
            except Exception as e:
                self._log(f"❌ Error in C test generation (attempt {attempt + 1}): {e}", "ERROR")
                if attempt == max_retries - 1:
                    return ""
        
        self._log("❌ Failed to generate valid C test code after all retries", "ERROR")
        return ""
    
    def _basic_c_validation(self, code: str) -> bool:
        """Basic validation of C code structure"""
        # Check for basic C structure
        has_includes = "#include" in code
        has_main = "int main" in code or "void main" in code
        has_functions = "void test_" in code or "static void" in code
        balanced_braces = code.count('{') == code.count('}')
        has_cmocka = "cmocka" in code.lower() or "assert_" in code
        
        return has_includes and has_main and has_functions and balanced_braces and has_cmocka

    def _generate_c_test_suite(self, functions: List[CFunctionInfo]) -> str:
        """Generate C test suite with validation"""
        if not functions:
            return ""

        # Build comprehensive prompt
        prompt = self._build_c_test_prompt(functions)
        
        # Generate test code with retries
        test_code = self._invoke_llm_for_c_generation(prompt)
        
        if not test_code:
            return ""
        
        return test_code

    def _compile_and_run_tests(self, test_file_path: Path, changed_files: List[str]) -> Dict[str, Any]:
        """Compile and execute C tests"""
        self._log(f"🔨 Compiling C tests from {test_file_path}", "INFO")
        
        # Prepare compilation
        include_dirs = [str(self.project_root)]
        
        # Find additional include directories
        for file_path in changed_files:
            file_path_obj = self.project_root / file_path
            if file_path_obj.exists() and file_path_obj.suffix in ['.h', '.c']:
                include_dirs.append(str(file_path_obj.parent))
        
        # Remove duplicates
        include_dirs = list(set(include_dirs))
        
        # Compile test executable
        test_executable = test_file_path.with_suffix('')
        
        compile_cmd = [
            'gcc', '-std=c99', '-Wall', '-Wextra', '-g',
            str(test_file_path),
            '-lcmocka',  # Link CMocka library
            '-o', str(test_executable)
        ]
        
        # Add include directories
        for inc_dir in include_dirs:
            compile_cmd.extend(['-I', inc_dir])
        
        try:
            # Compile
            self._log(f"🔧 Compilation command: {' '.join(compile_cmd)}", "INFO")
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True, text=True,
                cwd=str(self.project_root)
            )
            
            if compile_result.returncode != 0:
                self._log("❌ Compilation failed", "ERROR")
                self._log(compile_result.stderr, "ERROR")
                return {
                    "status": "failure",
                    "output": f"Compilation failed: {compile_result.stderr}",
                    "test_cases": []
                }
            
            self._log("✅ Compilation successful", "SUCCESS")
            
            # Run tests
            self._log(f"🧪 Executing C tests: {test_executable}", "INFO")
            run_result = subprocess.run(
                [str(test_executable)],
                capture_output=True, text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.project_root)
            )
            
            # Parse test results
            test_cases = self._parse_cmocka_output(run_result.stdout, run_result.stderr)
            
            if run_result.returncode == 0:
                self._log("✅ All tests passed", "SUCCESS")
                return {
                    "status": "success",
                    "output": run_result.stdout,
                    "stderr": run_result.stderr,
                    "test_cases": test_cases
                }
            else:
                self._log("❌ Some tests failed", "ERROR")
                self._log(run_result.stdout, "ERROR")
                self._log(run_result.stderr, "ERROR")
                return {
                    "status": "failure",
                    "output": run_result.stdout + "\n" + run_result.stderr,
                    "test_cases": test_cases
                }
                
        except subprocess.TimeoutExpired:
            self._log("❌ Test execution timed out", "ERROR")
            return {"status": "failure", "output": "Test execution timed out", "test_cases": []}
        except Exception as e:
            self._log(f"❌ Test execution failed: {e}", "ERROR")
            return {"status": "failure", "output": f"Execution error: {str(e)}", "test_cases": []}
    
    def _parse_cmocka_output(self, stdout: str, stderr: str) -> List[TestCaseResult]:
        """Parse CMocka test output"""
        test_cases = []
        full_output = stdout + "\n" + stderr
        
        # Look for CMocka test patterns
        # CMocka typically outputs: [ RUN      ] test_name
        #                          [       OK ] test_name
        #                          [  FAILED  ] test_name
        
        run_pattern = re.compile(r'\[\s*RUN\s*\]\s+(\w+)')
        ok_pattern = re.compile(r'\[\s*OK\s*\]\s+(\w+)')
        failed_pattern = re.compile(r'\[\s*FAILED\s*\]\s+(\w+)')
        
        running_tests = set()
        
        for match in run_pattern.finditer(full_output):
            test_name = match.group(1)
            running_tests.add(test_name)
        
        for match in ok_pattern.finditer(full_output):
            test_name = match.group(1)
            test_cases.append(TestCaseResult(
                name=test_name,
                status="PASS",
                test_method=test_name
            ))
            running_tests.discard(test_name)
        
        for match in failed_pattern.finditer(full_output):
            test_name = match.group(1)
            test_cases.append(TestCaseResult(
                name=test_name,
                status="FAIL",
                test_method=test_name,
                failure_reason="Test assertion failed"
            ))
            running_tests.discard(test_name)
        
        # Any remaining running tests are errors
        for test_name in running_tests:
            test_cases.append(TestCaseResult(
                name=test_name,
                status="ERROR",
                test_method=test_name,
                error_message="Test did not complete"
            ))
        
        return test_cases

    def _generate_json_report(self, report: TestReport) -> str:
        """Generate JSON report for C project"""
        test_cases_dict = [tc.to_dict() for tc in report.test_cases]
        
        clean_test_results = dict(report.test_results)
        if 'test_cases' in clean_test_results:
            clean_test_results['test_cases'] = [tc.to_dict() if hasattr(tc, 'to_dict') else tc for tc in clean_test_results['test_cases']]
        
        report_dict = {
            "title": "🚀 C Project Test Automation Report",
            "generated": report.timestamp,
            "model_name": report.model_name,
            "execution_time_seconds": report.execution_time,
            "status": report.status,
            "changed_files": report.changed_files,
            "compilation_status": report.compilation_status,
            "analyzed_functions": [
                {
                    "name": func.name,
                    "file_path": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "return_type": func.signature.return_type,
                    "parameters": func.signature.parameters,
                    "is_static": func.signature.is_static,
                    "is_extern": func.signature.is_extern,
                    "complexity_score": func.complexity_score,
                    "code_snippet": func.code[:200] + "..." if len(func.code) > 200 else func.code
                }
                for func in report.analyzed_functions
            ],
            "test_results": clean_test_results,
            "test_cases": test_cases_dict,
            "coverage_metrics": report.coverage_metrics,
            "execution_logs": report.logs
        }
        
        return json.dumps(report_dict, indent=2, ensure_ascii=False)

    def _generate_xml_report(self, report: TestReport) -> str:
        """Generate XML report for C project"""
        root = ET.Element("c_test_automation_report")
        
        # Header
        header = ET.SubElement(root, "header")
        ET.SubElement(header, "title").text = "🚀 C Project Test Automation Report"
        ET.SubElement(header, "generated").text = report.timestamp
        ET.SubElement(header, "model_name").text = report.model_name
        ET.SubElement(header, "execution_time_seconds").text = str(report.execution_time)
        ET.SubElement(header, "status").text = report.status
        
        # Compilation status
        comp_elem = ET.SubElement(root, "compilation_status")
        for key, value in report.compilation_status.items():
            comp_test_elem = ET.SubElement(comp_elem, "file")
            comp_test_elem.set("name", key)
            comp_test_elem.text = str(value)
        
        # Functions
        functions_elem = ET.SubElement(root, "analyzed_functions")
        ET.SubElement(functions_elem, "count").text = str(len(report.analyzed_functions))
        for func in report.analyzed_functions:
            func_elem = ET.SubElement(functions_elem, "function")
            ET.SubElement(func_elem, "name").text = func.name
            ET.SubElement(func_elem, "file_path").text = func.file_path
            ET.SubElement(func_elem, "return_type").text = func.signature.return_type
            ET.SubElement(func_elem, "complexity_score").text = str(func.complexity_score)
            ET.SubElement(func_elem, "is_static").text = str(func.signature.is_static)
        
        # Test results
        test_results_elem = ET.SubElement(root, "test_results")
        ET.SubElement(test_results_elem, "status").text = report.test_results.get("status", "unknown")
        if "output" in report.test_results:
            ET.SubElement(test_results_elem, "output").text = report.test_results["output"]
        
        # Test cases
        test_cases_elem = ET.SubElement(root, "test_cases")
        for test_case in report.test_cases:
            case_elem = ET.SubElement(test_cases_elem, "test_case")
            case_elem.set("name", test_case.name)
            case_elem.set("status", test_case.status)
            if test_case.failure_reason:
                ET.SubElement(case_elem, "failure_reason").text = test_case.failure_reason
        
        return ET.tostring(root, encoding='unicode', method='xml')

    def _generate_text_report(self, report: TestReport) -> str:
        """Generate human-readable text report for C project"""
        lines = [
            "🚀 C Project Test Automation Report",
            f"Generated: {report.timestamp}",
            f"Model Used: {report.model_name}",
            f"Execution Time: {report.execution_time:.2f} seconds",
            f"Status: {report.status}",
            "",
            "=" * 60,
            "",
            "📁 CHANGED FILES:",
            ""
        ]
        
        for i, file_path in enumerate(report.changed_files, 1):
            lines.append(f"  {i}. {file_path}")
        
        # Compilation status
        if report.compilation_status:
            lines.extend([
                "",
                "🔨 COMPILATION STATUS:",
                ""
            ])
            for file_name, compiled in report.compilation_status.items():
                status = "✅ SUCCESS" if compiled else "❌ FAILED"
                lines.append(f"  {file_name}: {status}")
        
        lines.extend([
            "",
            f"🔍 ANALYZED C FUNCTIONS ({len(report.analyzed_functions)} total):",
            ""
        ])
        
        for i, func in enumerate(report.analyzed_functions, 1):
            storage = ""
            if func.signature.is_static:
                storage += "static "
            if func.signature.is_extern:
                storage += "extern "
            
            param_count = len(func.signature.parameters)
            param_info = f" ({param_count} params)" if param_count > 0 else " (void)"
            
            lines.append(f"  {i}. {storage}{func.signature.return_type} {func.name}{param_info} [Complexity: {func.complexity_score}/10]")
            lines.append(f"     File: {func.file_path} (lines {func.line_start}-{func.line_end})")
            lines.append("")
        
        lines.extend([
            "🧪 TEST RESULTS:",
            f"  Status: {report.test_results.get('status', 'unknown').upper()}",
            ""
        ])
        
        # Individual test cases
        if report.test_cases:
            lines.extend([
                "📋 INDIVIDUAL TEST CASES:",
                ""
            ])
            
            for i, test_case in enumerate(report.test_cases, 1):
                status_emoji = "✅" if test_case.status == "PASS" else "❌" if test_case.status in ["FAIL", "ERROR"] else "⚠️"
                lines.append(f"  {i}. {test_case.name}")
                lines.append(f"     Status: {status_emoji} {test_case.status}")
                
                if test_case.failure_reason and test_case.status in ["FAIL", "ERROR"]:
                    lines.append(f"     Reason: {test_case.failure_reason}")
                
                lines.append("")
        
        if "output" in report.test_results and report.test_results["output"]:
            lines.extend([
                "  Test Output:",
                "  " + "─" * 40,
                *[f"  {line}" for line in report.test_results["output"].split('\n')[:15]],
                "  " + "─" * 40,
                ""
            ])
        
        lines.extend([
            "📝 EXECUTION LOGS (Last 25 entries):",
            ""
        ])
        
        for log_entry in report.logs[-25:]:
            lines.append(f"  {log_entry}")
        
        lines.extend([
            "",
            "=" * 60,
            f"Report completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return '\n'.join(lines)

    def _save_reports(self, report: TestReport):
        """Save reports in multiple formats"""
        timestamp_str = report.timestamp.replace(":", "-").replace(" ", "_")
        base_filename = f"c_test_automation_report_{timestamp_str.split('_')[0]}_{timestamp_str.split('_')[1]}"
        
        # Save JSON report
        json_content = self._generate_json_report(report)
        json_path = self.reports_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        self._log(f"📄 JSON report saved: {json_path}", "INFO")
        
        # Save XML report
        xml_content = self._generate_xml_report(report)
        xml_path = self.reports_dir / f"{base_filename}.xml"
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        self._log(f"📄 XML report saved: {xml_path}", "INFO")
        
        # Save text report
        text_content = self._generate_text_report(report)
        text_path = self.reports_dir / f"{base_filename}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        self._log(f"📄 Text report saved: {text_path}", "INFO")

    def run(self):
        """Main runner for C project testing"""
        self._log("🚀 Starting C Project Test Automation", "INFO")
        
        changed_files = self._get_changed_files()
        if not changed_files:
            self._log("No C files changed. Exiting CI run.", "INFO")
            return

        all_changed_functions = []
        compilation_status = {}
        
        for file_path_str in changed_files:
            file_path = self.project_root / file_path_str
            if file_path.exists() and file_path.suffix in ['.c', '.h']:
                self._log(f"🔍 Analyzing C file: {file_path_str}", "INFO")
                
                # Only extract functions from .c files, not headers
                if file_path.suffix == '.c':
                    try:
                        functions = self._extract_functions_from_c_file(file_path)
                        all_changed_functions.extend(functions)
                        compilation_status[file_path_str] = True
                        self._log(f"✅ Found {len(functions)} functions in {file_path_str}", "INFO")
                        
                    except Exception as e:
                        compilation_status[file_path_str] = False
                        self._log(f"❌ Error analyzing {file_path_str}: {e}", "ERROR")
                else:
                    # For .h files, just mark as analyzed
                    compilation_status[file_path_str] = True
                    self._log(f"📄 Header file analyzed: {file_path_str}", "INFO")

        if not all_changed_functions:
            self._log("No C functions found in changed files. Exiting.", "WARNING")
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            report = TestReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=self.model_name,
                changed_files=changed_files,
                analyzed_functions=all_changed_functions,
                test_results={"status": "skipped", "output": "No C functions found"},
                coverage_metrics={},
                status="SKIPPED",
                execution_time=execution_time,
                logs=self.logs,
                compilation_status=compilation_status
            )
            self._save_reports(report)
            return

        # Function analysis summary
        total_functions = len(all_changed_functions)
        static_functions = sum(1 for f in all_changed_functions if f.signature.is_static)
        avg_complexity = sum(f.complexity_score for f in all_changed_functions) / total_functions
        
        self._log(f"📊 C Function Analysis Summary:", "INFO")
        self._log(f"   Total Functions: {total_functions}", "INFO")
        self._log(f"   Static Functions: {static_functions}", "INFO")
        self._log(f"   Average Complexity: {avg_complexity:.1f}/10", "INFO")
        
        # Generate C test suite
        self._log(f"🧠 Generating C test cases using {self.model_name}...", "INFO")
        test_content = self._generate_c_test_suite(all_changed_functions)
        
        if not test_content:
            self._log("❌ Failed to generate C test cases. Aborting.", "ERROR")
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            report = TestReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=self.model_name,
                changed_files=changed_files,
                analyzed_functions=all_changed_functions,
                test_results={"status": "failure", "output": "Failed to generate C test cases"},
                coverage_metrics={},
                status="FAILED",
                execution_time=execution_time,
                logs=self.logs,
                compilation_status=compilation_status
            )
            self._save_reports(report)
            return

        # Save C test file
        test_file_name = "pr_generated_tests.c"
        test_file_path = self.test_dir / test_file_name
        
        try:
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            compilation_status['generated_test_file'] = True
            self._log(f"✅ Generated and saved C tests to {test_file_path}", "SUCCESS")
                
        except Exception as e:
            compilation_status['generated_test_file'] = False
            self._log(f"❌ Failed to save C test file: {e}", "ERROR")
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            report = TestReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=self.model_name,
                changed_files=changed_files,
                analyzed_functions=all_changed_functions,
                test_results={"status": "failure", "output": f"Failed to save test file: {e}"},
                coverage_metrics={},
                status="FAILED",
                execution_time=execution_time,
                logs=self.logs,
                compilation_status=compilation_status
            )
            self._save_reports(report)
            return

        # Compile and run C tests
        self._log("🏃 Compiling and executing C tests...", "INFO")
        test_results = self._compile_and_run_tests(test_file_path, changed_files)
        
        # Calculate execution time and determine status
        execution_time = (datetime.now() - self.start_time).total_seconds()
        overall_status = "SUCCESS" if test_results["status"] == "success" else "FAILED"
        
        # Create comprehensive report
        report = TestReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name=self.model_name,
            changed_files=changed_files,
            analyzed_functions=all_changed_functions,
            test_results=test_results,
            coverage_metrics={},  # C coverage would need gcov integration
            status=overall_status,
            execution_time=execution_time,
            logs=self.logs,
            compilation_status=compilation_status,
            test_cases=test_results.get("test_cases", [])
        )
        
        # Save reports
        self._save_reports(report)
        
        # Final summary
        self._log(f"✨ C test automation completed in {execution_time:.2f} seconds", "INFO")
        self._log(f"📊 Final Status: {overall_status}", "INFO")
        self._log(f"🔧 Compilation Status: {sum(compilation_status.values())}/{len(compilation_status)} files successful", "INFO")


if __name__ == "__main__":
    try:
        runner = CChangeAnalyzerAndTester()
        runner.run()
    except Exception as e:
        print(f"❌ Critical error in C test automation: {e}")
        print("📊 Attempting to save error report...")
        
        try:
            from datetime import datetime
            error_report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "CRITICAL_ERROR",
                "error": str(e),
                "message": "C test automation failed with critical error"
            }
            
            import os
            os.makedirs("reports", exist_ok=True)
            
            import json
            with open("reports/c_critical_error_report.json", "w") as f:
                json.dump(error_report, f, indent=2)
            
            print("📊 Critical error report saved to reports/c_critical_error_report.json")
        except Exception as save_error:
            print(f"❌ Could not save error report: {save_error}")
        
        exit(0)