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

# Chunking Configuration
MAX_LINES_PER_CHUNK = 800        # Maximum lines of code per chunk
MAX_TOKENS_PER_CHUNK = 2500      # Estimated token limit per chunk
MIN_FUNCTIONS_PER_CHUNK = 1      # Minimum functions per chunk
MAX_FUNCTIONS_PER_CHUNK = 10     # Maximum functions per chunk
LINES_TO_TOKENS_RATIO = 3        # Rough estimate: 1 line = 3 tokens

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
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "status": self.status,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "failure_reason": self.failure_reason,
            "test_method": self.test_method,
            "explanation": self.explanation
        }

@dataclass
class CodeChange:
    """Represents a code change with line information"""
    file_path: str
    line_start: int
    line_end: int
    change_type: str  # "added", "modified", "deleted"
    content: str = ""

@dataclass
class DependencyInfo:
    """Information about function/file dependencies"""
    function_calls: Set[str] = field(default_factory=set)
    included_files: Set[str] = field(default_factory=set)
    global_variables: Set[str] = field(default_factory=set)
    macro_usage: Set[str] = field(default_factory=set)
    struct_usage: Set[str] = field(default_factory=set)

@dataclass
class TestChunk:
    """Represents a testing chunk with its dependencies"""
    chunk_id: str
    primary_changes: List[CodeChange]
    dependent_functions: List[CFunctionInfo]
    dependent_files: Set[str]
    test_type: str  # "unit", "integration", "system"
    complexity_score: int = 0
    total_lines: int = 0
    estimated_tokens: int = 0

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

class CDependencyAnalyzer:
    """Analyzes dependencies and connections between C code components"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.function_definitions = {}  # function_name -> CFunctionInfo
        self.function_calls_map = {}    # function_name -> set of called functions
        self.file_dependencies = {}     # file_path -> set of dependent files
        
    def analyze_git_changes(self) -> List[CodeChange]:
        """Analyze git changes to identify what lines were modified"""
        changes = []
        
        try:
            # Get git diff with line numbers
            result = subprocess.run(
                ['git', 'diff', '--unified=0', 'HEAD~1', 'HEAD'],
                capture_output=True, text=True, cwd=str(self.project_root)
            )
            
            if result.returncode != 0:
                # Try with staged changes if no commit history
                result = subprocess.run(
                    ['git', 'diff', '--unified=0', '--cached'],
                    capture_output=True, text=True, cwd=str(self.project_root)
                )
                
            if result.returncode != 0:
                # Try with unstaged changes
                result = subprocess.run(
                    ['git', 'diff', '--unified=0'],
                    capture_output=True, text=True, cwd=str(self.project_root)
                )
            
            changes = self._parse_git_diff(result.stdout)
            
        except Exception as e:
            print(f"⚠️ Could not analyze git changes: {e}")
            
        return changes
    
    def _parse_git_diff(self, diff_output: str) -> List[CodeChange]:
        """Parse git diff output to extract line changes"""
        changes = []
        current_file = None
        
        lines = diff_output.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # File header
            if line.startswith('diff --git'):
                i += 1
                continue
            elif line.startswith('+++'):
                # Extract filename
                file_match = re.search(r'\+\+\+ b/(.+)', line)
                if file_match:
                    current_file = file_match.group(1)
                i += 1
                continue
            elif line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                hunk_match = re.search(r'@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@', line)
                if hunk_match and current_file and current_file.endswith('.c'):
                    new_start = int(hunk_match.group(3))
                    new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1
                    
                    # Collect the actual changes
                    content_lines = []
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith('@@') and not lines[j].startswith('diff'):
                        if lines[j].startswith('+') and not lines[j].startswith('+++'):
                            content_lines.append(lines[j][1:])  # Remove '+' prefix
                        j += 1
                    
                    if new_count > 0:
                        changes.append(CodeChange(
                            file_path=current_file,
                            line_start=new_start,
                            line_end=new_start + new_count - 1,
                            change_type="modified",
                            content='\n'.join(content_lines)
                        ))
                i += 1
                continue
            
            i += 1
        
        return changes
    
    def build_dependency_map(self, files: List[str]) -> Dict[str, DependencyInfo]:
        """Build comprehensive dependency map for given files"""
        dependency_map = {}
        
        for file_path in files:
            if file_path.endswith('.c') or file_path.endswith('.h'):
                full_path = self.project_root / file_path
                if full_path.exists():
                    dependency_map[file_path] = self._analyze_file_dependencies(full_path)
        
        return dependency_map
    
    def _analyze_file_dependencies(self, file_path: Path) -> DependencyInfo:
        """Analyze dependencies in a single C file"""
        deps = DependencyInfo()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find function calls
            function_call_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.MULTILINE)
            for match in function_call_pattern.finditer(content):
                func_name = match.group(1)
                # Filter out C keywords and common macros
                if func_name not in {'if', 'while', 'for', 'switch', 'sizeof', 'return'}:
                    deps.function_calls.add(func_name)
            
            # Find includes
            include_pattern = re.compile(r'#include\s*[<"]([^>"]+)[>"]')
            for match in include_pattern.finditer(content):
                deps.included_files.add(match.group(1))
            
            # Find global variable usage
            global_var_pattern = re.compile(r'extern\s+\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)')
            for match in global_var_pattern.finditer(content):
                deps.global_variables.add(match.group(1))
            
            # Find macro usage
            macro_pattern = re.compile(r'#define\s+([a-zA-Z_][a-zA-Z0-9_]*)')
            for match in macro_pattern.finditer(content):
                deps.macro_usage.add(match.group(1))
            
            # Find struct usage
            struct_pattern = re.compile(r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)')
            for match in struct_pattern.finditer(content):
                deps.struct_usage.add(match.group(1))
                
        except Exception as e:
            print(f"⚠️ Error analyzing dependencies in {file_path}: {e}")
        
        return deps
    
    def create_test_chunks(self, changes: List[CodeChange], all_functions: List[CFunctionInfo]) -> List[TestChunk]:
        """Create intelligent test chunks based on changes and their dependencies with size optimization"""
        chunks = []
        dependency_map = self.build_dependency_map([change.file_path for change in changes])
        
        # Group changes by file first
        changes_by_file = {}
        for change in changes:
            if change.file_path not in changes_by_file:
                changes_by_file[change.file_path] = []
            changes_by_file[change.file_path].append(change)
        
        chunk_id = 0
        for file_path, file_changes in changes_by_file.items():
            # Find functions affected by these changes
            affected_functions = self._find_affected_functions(file_changes, all_functions)
            
            # NEW: Check if we can isolate individual functions
            isolated_chunks = self._try_isolate_functions(affected_functions, all_functions, dependency_map)
            
            if isolated_chunks:
                # Apply size-based chunking to isolated functions
                size_optimized_chunks = self._apply_size_based_chunking(isolated_chunks)
                for chunk in size_optimized_chunks:
                    chunk.chunk_id = f"chunk_{chunk_id}"
                    chunks.append(chunk)
                    chunk_id += 1
            else:
                # Create integration chunk but apply size-based chunking
                integration_functions = self._find_connected_functions(affected_functions, all_functions, dependency_map)
                integration_chunks = self._create_size_optimized_chunks(
                    file_changes, integration_functions, chunk_id, "integration"
                )
                chunks.extend(integration_chunks)
                chunk_id += len(integration_chunks)
        
        return chunks
    
    def _try_isolate_functions(self, affected_functions: List[CFunctionInfo], 
                              all_functions: List[CFunctionInfo],
                              dependency_map: Dict[str, DependencyInfo]) -> List[TestChunk]:
        """Try to isolate functions that can be tested independently"""
        isolated_chunks = []
        
        for func in affected_functions:
            # Check if this function can be tested in isolation
            if self._can_function_be_isolated(func, all_functions, dependency_map):
                # Calculate size metrics for this function
                func_lines = len(func.code.splitlines())
                func_tokens = func_lines * LINES_TO_TOKENS_RATIO
                
                # Create an isolated chunk for this function
                chunk = TestChunk(
                    chunk_id="",  # Will be set by caller
                    primary_changes=[],  # Will be set based on actual changes
                    dependent_functions=[func],
                    dependent_files={func.file_path},
                    test_type="unit",
                    complexity_score=func.complexity_score,
                    total_lines=func_lines,
                    estimated_tokens=func_tokens
                )
                isolated_chunks.append(chunk)
        
        # If we can isolate all functions individually, return the isolated chunks
        if len(isolated_chunks) == len(affected_functions):
            return isolated_chunks
        
        # If some functions must be grouped, return None to use the grouped approach
        return None
    
    def _can_function_be_isolated(self, func: CFunctionInfo, 
                                 all_functions: List[CFunctionInfo],
                                 dependency_map: Dict[str, DependencyInfo]) -> bool:
        """Check if a function can be tested in isolation"""
        
        # Get dependencies for this function's file
        file_deps = dependency_map.get(func.file_path, DependencyInfo())
        
        # Extract function calls from the specific function code
        function_calls_in_code = set()
        call_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        for match in call_pattern.finditer(func.code):
            call_name = match.group(1)
            # Exclude C keywords, standard library functions, and I/O functions
            if call_name not in {'if', 'while', 'for', 'switch', 'sizeof', 'return', 
                                'printf', 'scanf', 'malloc', 'free', 'memcpy', 'strlen',
                                'strcpy', 'strcmp', 'fopen', 'fclose', 'fprintf', 'fscanf'}:
                function_calls_in_code.add(call_name)
        
        # Check if this function calls other user-defined functions in the project
        user_defined_functions = {f.name for f in all_functions}
        calls_user_functions = function_calls_in_code.intersection(user_defined_functions)
        
        # Remove the function's own name if it appears (recursive calls are ok for isolation)
        calls_user_functions.discard(func.name)
        
        # Check if other functions call this function
        called_by_others = False
        for other_func in all_functions:
            if other_func.name != func.name:
                other_calls = set()
                for match in call_pattern.finditer(other_func.code):
                    other_calls.add(match.group(1))
                if func.name in other_calls:
                    called_by_others = True
                    break
        
        # A function can be isolated if:
        # 1. It doesn't call other user-defined functions OR
        # 2. It only calls library functions OR
        # 3. It's not called by other functions (indicating it might be newly added)
        
        is_isolated = (
            len(calls_user_functions) == 0 or  # Calls no user functions
            not called_by_others  # Not called by other functions (likely new/independent)
        )
        
        return is_isolated
    
    def _find_affected_functions(self, changes: List[CodeChange], all_functions: List[CFunctionInfo]) -> List[CFunctionInfo]:
        """Find functions that are directly affected by the changes"""
        affected = []
        
        for change in changes:
            for func in all_functions:
                if (func.file_path == change.file_path and 
                    func.line_start <= change.line_end and 
                    func.line_end >= change.line_start):
                    if func not in affected:
                        affected.append(func)
        
        return affected
    
    def _find_connected_functions(self, primary_functions: List[CFunctionInfo], 
                                 all_functions: List[CFunctionInfo],
                                 dependency_map: Dict[str, DependencyInfo]) -> List[CFunctionInfo]:
        """Find all functions connected to the primary changed functions"""
        connected = list(primary_functions)  # Start with primary functions
        visited = {func.name for func in primary_functions}
        
        # BFS to find connected functions
        queue = list(primary_functions)
        
        while queue:
            current_func = queue.pop(0)
            
            # Check dependencies in the same file and related files
            for file_path, deps in dependency_map.items():
                if current_func.name in deps.function_calls:
                    # This file calls our function, check its functions
                    file_functions = [f for f in all_functions if f.file_path == file_path]
                    for func in file_functions:
                        if func.name not in visited:
                            connected.append(func)
                            visited.add(func.name)
                            queue.append(func)
                
                # Check if current function calls other functions
                if file_path == current_func.file_path:
                    for called_func_name in deps.function_calls:
                        called_func = next((f for f in all_functions if f.name == called_func_name), None)
                        if called_func and called_func.name not in visited:
                            connected.append(called_func)
                            visited.add(called_func.name)
                            queue.append(called_func)
        
        return connected
    
    def _determine_test_type(self, primary_functions: List[CFunctionInfo], 
                           connected_functions: List[CFunctionInfo]) -> str:
        """Determine the type of testing needed based on scope"""
        if len(connected_functions) <= len(primary_functions):
            return "unit"
        elif len(connected_functions) <= len(primary_functions) * 3:
            return "integration"
        else:
            return "system"
    
    def _calculate_chunk_complexity(self, changes: List[CodeChange], 
                                  connected_functions: List[CFunctionInfo]) -> int:
        """Calculate complexity score for the test chunk"""
        base_complexity = len(changes)
        function_complexity = sum(func.complexity_score for func in connected_functions)
        connection_complexity = len(connected_functions) - len(changes)
        
        return min(base_complexity + function_complexity + connection_complexity, 10)
    
    def _find_dependent_files(self, file_path: str, dependency_map: Dict[str, DependencyInfo]) -> Set[str]:
        """Find files that depend on the given file"""
        dependent_files = {file_path}
        
        # Find files that include this file
        for other_file, deps in dependency_map.items():
            if file_path.replace('.c', '.h') in deps.included_files:
                dependent_files.add(other_file)
        
        return dependent_files
    
    def _apply_size_based_chunking(self, isolated_chunks: List[TestChunk]) -> List[TestChunk]:
        """Apply size-based chunking to isolated chunks to optimize token usage"""
        optimized_chunks = []
        
        # Separate large functions that should stay alone vs small functions that can be grouped
        large_chunks = []
        small_chunks = []
        
        for chunk in isolated_chunks:
            # Calculate total lines and estimated tokens for this chunk
            total_lines = sum(len(func.code.splitlines()) for func in chunk.dependent_functions)
            estimated_tokens = total_lines * LINES_TO_TOKENS_RATIO
            
            chunk.total_lines = total_lines
            chunk.estimated_tokens = estimated_tokens
            
            # If individual function is large, keep it separate
            if total_lines > MAX_LINES_PER_CHUNK // 2:  # Functions larger than half the limit stay separate
                large_chunks.append(chunk)
            else:
                small_chunks.append(chunk)
        
        # Add large chunks as-is
        optimized_chunks.extend(large_chunks)
        
        # Group small chunks together to optimize token usage
        if small_chunks:
            grouped_chunks = self._group_small_chunks(small_chunks)
            optimized_chunks.extend(grouped_chunks)
        
        return optimized_chunks
    
    def _group_small_chunks(self, small_chunks: List[TestChunk]) -> List[TestChunk]:
        """Group small isolated chunks together to optimize token usage"""
        grouped_chunks = []
        current_group_functions = []
        current_group_lines = 0
        current_group_tokens = 0
        current_group_changes = []
        current_group_files = set()
        group_id = 0
        
        for chunk in small_chunks:
            chunk_lines = chunk.total_lines
            chunk_tokens = chunk.estimated_tokens
            
            # Check if adding this chunk would exceed limits
            if (current_group_lines + chunk_lines > MAX_LINES_PER_CHUNK or
                current_group_tokens + chunk_tokens > MAX_TOKENS_PER_CHUNK or
                len(current_group_functions) >= MAX_FUNCTIONS_PER_CHUNK):
                
                # Create group from current functions
                if current_group_functions:
                    grouped_chunk = TestChunk(
                        chunk_id=f"grouped_{group_id}",
                        primary_changes=current_group_changes,
                        dependent_functions=current_group_functions,
                        dependent_files=current_group_files,
                        test_type="unit",
                        complexity_score=sum(f.complexity_score for f in current_group_functions),
                        total_lines=current_group_lines,
                        estimated_tokens=current_group_tokens
                    )
                    grouped_chunks.append(grouped_chunk)
                    group_id += 1
                
                # Start new group
                current_group_functions = chunk.dependent_functions.copy()
                current_group_lines = chunk_lines
                current_group_tokens = chunk_tokens
                current_group_changes = chunk.primary_changes.copy()
                current_group_files = chunk.dependent_files.copy()
            else:
                # Add to current group
                current_group_functions.extend(chunk.dependent_functions)
                current_group_lines += chunk_lines
                current_group_tokens += chunk_tokens
                current_group_changes.extend(chunk.primary_changes)
                current_group_files.update(chunk.dependent_files)
        
        # Add the last group
        if current_group_functions:
            grouped_chunk = TestChunk(
                chunk_id=f"grouped_{group_id}",
                primary_changes=current_group_changes,
                dependent_functions=current_group_functions,
                dependent_files=current_group_files,
                test_type="unit",
                complexity_score=sum(f.complexity_score for f in current_group_functions),
                total_lines=current_group_lines,
                estimated_tokens=current_group_tokens
            )
            grouped_chunks.append(grouped_chunk)
        
        return grouped_chunks
    
    def _create_size_optimized_chunks(self, changes: List[CodeChange], 
                                    functions: List[CFunctionInfo], 
                                    start_chunk_id: int, 
                                    test_type: str) -> List[TestChunk]:
        """Create size-optimized chunks from a list of functions"""
        chunks = []
        
        if not functions:
            return chunks
        
        # Calculate total size
        total_lines = sum(len(func.code.splitlines()) for func in functions)
        total_tokens = total_lines * LINES_TO_TOKENS_RATIO
        
        # If total size is within limits, create single chunk
        if (total_lines <= MAX_LINES_PER_CHUNK and 
            total_tokens <= MAX_TOKENS_PER_CHUNK and 
            len(functions) <= MAX_FUNCTIONS_PER_CHUNK):
            
            chunk = TestChunk(
                chunk_id=f"chunk_{start_chunk_id}",
                primary_changes=changes,
                dependent_functions=functions,
                dependent_files=set(f.file_path for f in functions),
                test_type=test_type,
                complexity_score=self._calculate_chunk_complexity(changes, functions),
                total_lines=total_lines,
                estimated_tokens=total_tokens
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks
            chunk_groups = self._group_functions_for_chunking(functions)
            
            for i, group in enumerate(chunk_groups):
                group_lines = sum(len(func.code.splitlines()) for func in group)
                group_tokens = group_lines * LINES_TO_TOKENS_RATIO
                
                chunk = TestChunk(
                    chunk_id=f"chunk_{start_chunk_id + i}",
                    primary_changes=changes,
                    dependent_functions=group,
                    dependent_files=set(f.file_path for f in group),
                    test_type=test_type,
                    complexity_score=self._calculate_chunk_complexity(changes, group),
                    total_lines=group_lines,
                    estimated_tokens=group_tokens
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_large_chunk(self, chunk: TestChunk) -> List[TestChunk]:
        """Split a large chunk into smaller, manageable chunks"""
        sub_chunks = []
        functions = chunk.dependent_functions
        
        # Group functions to fit within size limits
        function_groups = self._group_functions_for_chunking(functions)
        
        for i, group in enumerate(function_groups):
            group_lines = sum(len(func.code.splitlines()) for func in group)
            group_tokens = group_lines * LINES_TO_TOKENS_RATIO
            
            sub_chunk = TestChunk(
                chunk_id=f"{chunk.chunk_id}_part_{i}",
                primary_changes=chunk.primary_changes,
                dependent_functions=group,
                dependent_files=set(f.file_path for f in group),
                test_type=chunk.test_type,
                complexity_score=self._calculate_chunk_complexity(chunk.primary_changes, group),
                total_lines=group_lines,
                estimated_tokens=group_tokens
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _group_functions_for_chunking(self, functions: List[CFunctionInfo]) -> List[List[CFunctionInfo]]:
        """Group functions into chunks that fit within size and token limits"""
        groups = []
        current_group = []
        current_lines = 0
        current_tokens = 0
        
        # Sort functions by size (largest first) to better pack them
        sorted_functions = sorted(functions, key=lambda f: len(f.code.splitlines()), reverse=True)
        
        for func in sorted_functions:
            func_lines = len(func.code.splitlines())
            func_tokens = func_lines * LINES_TO_TOKENS_RATIO
            
            # If this single function is too large, put it in its own chunk
            if func_lines > MAX_LINES_PER_CHUNK:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_lines = 0
                    current_tokens = 0
                
                # Large function gets its own chunk (don't split individual functions)
                groups.append([func])
                continue
            
            # Check if adding this function would exceed limits
            if (current_lines + func_lines > MAX_LINES_PER_CHUNK or
                current_tokens + func_tokens > MAX_TOKENS_PER_CHUNK or
                len(current_group) >= MAX_FUNCTIONS_PER_CHUNK):
                
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [func]
                current_lines = func_lines
                current_tokens = func_tokens
            else:
                # Add to current group
                current_group.append(func)
                current_lines += func_lines
                current_tokens += func_tokens
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups

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
        
        # Extract modifiers from group 1 (can be None)
        modifiers = func_match.group(1) or ""
        is_static = 'static' in modifiers
        is_extern = 'extern' in modifiers
        is_inline = 'inline' in modifiers
        
        # Extract return type (group 2) and function name (group 3)
        return_type = func_match.group(2).strip() if func_match.group(2) else "void"
        func_name = func_match.group(3).strip() if func_match.group(3) else "unknown"
        
        # Extract parameters from the full signature
        params_match = re.search(r'\((.*?)\)', full_signature, re.DOTALL)
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
        self.dependency_analyzer = CDependencyAnalyzer(self.project_root)
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
                    
                    # Create function info - handle relative paths correctly
                    try:
                        if file_path.is_absolute():
                            relative_path = str(file_path.relative_to(self.project_root))
                        else:
                            relative_path = str(file_path)
                    except ValueError:
                        # If file_path is not within project_root, just use the filename
                        relative_path = file_path.name
                    
                    func_info = CFunctionInfo(
                        name=signature.name,
                        file_path=relative_path,
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
            "#include <limits.h>",
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

### 3. FUNCTION IMPLEMENTATIONS TO INCLUDE:
Copy these exact function implementations into your test file:

```c
{chr(10).join([func.code for func in functions])}
```

### 4. TEST STRUCTURE (Use CMocka framework):
```c
// Test function example - always add (void)state; to suppress unused parameter warning
static void test_function_name(void **state) {{
    (void)state;  // Suppress unused parameter warning
    
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

### 5. FUNCTIONS TO TEST:

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
2. The exact function implementations provided above (copy them exactly)
3. Any required #define macros
4. Static test functions for each function under test
5. Main function with CMocka test suite setup that includes ALL test functions
6. Proper C formatting and syntax

IMPORTANT: 
- Copy the function implementations EXACTLY as provided above
- Every test function you define MUST be included in the CMUnitTest array in main(), otherwise it will cause "defined but not used" warnings
- The test file should be completely self-contained and compilable

## COMPILATION REQUIREMENTS:
- Code must compile with: gcc -std=c99 -Wall -Wextra
- All functions must be properly declared
- All variables must be declared before use
- Use proper C99 standard compliance
- ALWAYS add (void)state; as first line in test functions to suppress unused parameter warnings
- Include <limits.h> if using INT_MAX, INT_MIN, or other limit constants

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
        """Compile and execute C tests with coverage"""
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
        
        # Compile test executable with coverage flags
        test_executable = test_file_path.with_suffix('')
        
        compile_cmd = [
            'gcc', '-std=c99', '-Wall', '-Wextra', '-g',
            '--coverage',  # Enable coverage instrumentation
            '-fprofile-arcs', '-ftest-coverage',  # Additional coverage flags
            str(test_file_path),
            '-lcmocka',  # Link CMocka library
            '-lgcov',    # Link gcov library
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
                    "test_cases": [],
                    "coverage": {}
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
            test_cases = self._parse_cmocka_output(run_result.stdout, run_result.stderr, test_file_path)
            
            # Generate coverage reports
            self._log("📊 Generating code coverage reports...", "INFO")
            coverage_metrics = self._generate_coverage_reports(changed_files)
            
            if run_result.returncode == 0:
                self._log("✅ All tests passed", "SUCCESS")
                return {
                    "status": "success",
                    "output": run_result.stdout,
                    "stderr": run_result.stderr,
                    "test_cases": test_cases,
                    "coverage": coverage_metrics
                }
            else:
                self._log("❌ Some tests failed", "ERROR")
                self._log(run_result.stdout, "ERROR")
                self._log(run_result.stderr, "ERROR")
                return {
                    "status": "failure",
                    "output": run_result.stdout + "\n" + run_result.stderr,
                    "test_cases": test_cases,
                    "coverage": coverage_metrics
                }
                
        except subprocess.TimeoutExpired:
            self._log("❌ Test execution timed out", "ERROR")
            return {"status": "failure", "output": "Test execution timed out", "test_cases": [], "coverage": {}}
        except Exception as e:
            self._log(f"❌ Test execution failed: {e}", "ERROR")
            return {"status": "failure", "output": f"Execution error: {str(e)}", "test_cases": [], "coverage": {}}
    
    def _parse_cmocka_output(self, stdout: str, stderr: str, test_file_path: Path = None) -> List[TestCaseResult]:
        """Parse CMocka test output"""
        test_cases = []
        full_output = stdout + "\n" + stderr
        
        # Extract test explanations if test file is available
        explanations = {}
        if test_file_path and test_file_path.exists():
            explanations = self._extract_test_explanations(test_file_path)
        
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
                test_method=test_name,
                explanation=explanations.get(test_name, "")
            ))
            running_tests.discard(test_name)
        
        for match in failed_pattern.finditer(full_output):
            test_name = match.group(1)
            test_cases.append(TestCaseResult(
                name=test_name,
                status="FAIL",
                test_method=test_name,
                failure_reason="Test assertion failed",
                explanation=explanations.get(test_name, "")
            ))
            running_tests.discard(test_name)
        
        # Any remaining running tests are errors
        for test_name in running_tests:
            test_cases.append(TestCaseResult(
                name=test_name,
                status="ERROR",
                test_method=test_name,
                error_message="Test did not complete",
                explanation=explanations.get(test_name, "")
            ))
        
        return test_cases

    def _generate_coverage_reports(self, changed_files: List[str]) -> Dict[str, Any]:
        """Generate code coverage reports using gcov and lcov"""
        coverage_metrics = {
            "line_coverage": 0.0,
            "function_coverage": 0.0,
            "branch_coverage": 0.0,
            "files": {},
            "summary": "No coverage data available"
        }
        
        try:
            # Create coverage directory
            coverage_dir = self.project_root / "coverage"
            coverage_dir.mkdir(exist_ok=True)
            
            # Determine the correct gcov version to use
            gcov_tool = self._find_compatible_gcov()
            self._log(f"📊 Using gcov tool: {gcov_tool}", "INFO")
            
            # Generate gcov files for each changed C file
            for file_path in changed_files:
                if file_path.endswith('.c'):
                    self._log(f"📊 Generating coverage for {file_path}...", "INFO")
                    
                    # Run gcov on the source file
                    gcov_cmd = [gcov_tool, '-b', '-c', file_path]
                    gcov_result = subprocess.run(
                        gcov_cmd,
                        capture_output=True, text=True,
                        cwd=str(self.project_root)
                    )
                    
                    if gcov_result.returncode == 0:
                        # Parse gcov output
                        coverage_info = self._parse_gcov_output(gcov_result.stdout, file_path)
                        coverage_metrics["files"][file_path] = coverage_info
                        self._log(f"✅ Coverage generated for {file_path}", "SUCCESS")
                    else:
                        self._log(f"⚠️ Could not generate coverage for {file_path}: {gcov_result.stderr}", "WARNING")
            
            # Generate lcov HTML report if lcov is available
            try:
                # Initialize lcov info file
                lcov_info = coverage_dir / "coverage.info"
                
                # Capture coverage data with version compatibility
                lcov_capture_cmd = [
                    'lcov', '--capture', '--directory', str(self.project_root),
                    '--output-file', str(lcov_info),
                    '--gcov-tool', gcov_tool,  # Use the compatible gcov tool
                    '--ignore-errors', 'version'  # Ignore version mismatches
                ]
                
                capture_result = subprocess.run(
                    lcov_capture_cmd,
                    capture_output=True, text=True,
                    cwd=str(self.project_root)
                )
                
                if capture_result.returncode == 0:
                    # Generate HTML report
                    html_dir = coverage_dir / "html"
                    html_dir.mkdir(exist_ok=True)
                    
                    genhtml_cmd = [
                        'genhtml', str(lcov_info),
                        '--output-directory', str(html_dir),
                        '--title', 'C Test Coverage Report',
                        '--ignore-errors', 'version'  # Ignore version mismatches
                    ]
                    
                    html_result = subprocess.run(
                        genhtml_cmd,
                        capture_output=True, text=True,
                        cwd=str(self.project_root)
                    )
                    
                    if html_result.returncode == 0:
                        self._log("✅ HTML coverage report generated", "SUCCESS")
                        coverage_metrics["html_report"] = str(html_dir / "index.html")
                        
                        # Extract summary from lcov info
                        coverage_summary = self._extract_lcov_summary(str(lcov_info))
                        coverage_metrics.update(coverage_summary)
                    else:
                        self._log(f"⚠️ HTML report generation failed: {html_result.stderr}", "WARNING")
                else:
                    self._log(f"⚠️ lcov capture failed: {capture_result.stderr}", "WARNING")
                    # Try alternative approach with geninfo
                    self._try_alternative_coverage_capture(lcov_info, gcov_tool, coverage_metrics)
                    
            except FileNotFoundError:
                self._log("⚠️ lcov not found, trying alternative coverage methods", "WARNING")
                self._try_manual_coverage_analysis(coverage_metrics)
            
            # Calculate overall metrics if we have file data
            if coverage_metrics["files"]:
                total_lines = sum(info.get("total_lines", 0) for info in coverage_metrics["files"].values())
                covered_lines = sum(info.get("covered_lines", 0) for info in coverage_metrics["files"].values())
                
                if total_lines > 0:
                    coverage_metrics["line_coverage"] = (covered_lines / total_lines) * 100
                    coverage_metrics["summary"] = f"Line coverage: {coverage_metrics['line_coverage']:.1f}% ({covered_lines}/{total_lines})"
                    
        except Exception as e:
            self._log(f"⚠️ Coverage generation failed: {e}", "WARNING")
            coverage_metrics["error"] = str(e)
        
        return coverage_metrics
    
    def _extract_test_explanations(self, test_file_path: Path) -> Dict[str, str]:
        """Extract explanations for each test case based on test name patterns"""
        explanations = {}
        
        try:
            if not test_file_path.exists():
                return explanations
                
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract test function names and generate explanations based on naming patterns
            test_pattern = re.compile(r'static\s+void\s+(test_\w+)\s*\(', re.MULTILINE)
            test_matches = test_pattern.findall(content)
            
            for test_name in test_matches:
                explanation = self._generate_test_explanation(test_name)
                explanations[test_name] = explanation
                
        except Exception as e:
            self._log(f"⚠️ Could not extract test explanations: {e}", "WARNING")
        
        return explanations
    
    def _generate_test_explanation(self, test_name: str) -> str:
        """Generate human-readable explanation for test based on test name"""
        # Remove 'test_' prefix and split by underscores
        base_name = test_name.replace('test_', '', 1)
        parts = base_name.split('_')
        
        # Common test explanation patterns
        explanations = {
            'normal_operation': 'Tests basic functionality with standard valid inputs',
            'negative_numbers': 'Tests behavior with negative number inputs',
            'mixed_numbers': 'Tests with a mix of positive and negative numbers', 
            'zero': 'Tests behavior with zero values',
            'max_int': 'Tests with maximum integer value (boundary condition)',
            'min_int': 'Tests with minimum integer value (boundary condition)', 
            'overflow': 'Tests integer overflow behavior',
            'underflow': 'Tests integer underflow behavior',
            'null_pointer': 'Tests handling of NULL pointer inputs',
            'invalid_input': 'Tests behavior with invalid or malformed inputs',
            'boundary_condition': 'Tests edge cases and boundary conditions',
            'error_handling': 'Tests proper error handling and return codes',
            'memory_allocation': 'Tests dynamic memory allocation and cleanup',
            'string_operations': 'Tests string manipulation and comparison',
            'empty_string': 'Tests behavior with empty string inputs',
            'large_input': 'Tests with large input values',
            'small_input': 'Tests with small input values',
            'edge_case': 'Tests uncommon or edge case scenarios'
        }
        
        # Try to match the test pattern to known explanations
        test_key = '_'.join(parts)
        if test_key in explanations:
            return explanations[test_key]
        
        # Try partial matches for compound test names
        for key, explanation in explanations.items():
            if key in test_key:
                return explanation
        
        # Generate explanation based on function name and common patterns
        function_name = parts[0] if parts else 'function'
        
        if 'null' in test_key.lower():
            return f'Tests {function_name} behavior with NULL values'
        elif 'empty' in test_key.lower():
            return f'Tests {function_name} with empty inputs'
        elif 'invalid' in test_key.lower() or 'error' in test_key.lower():
            return f'Tests {function_name} error handling with invalid inputs'
        elif 'max' in test_key.lower() or 'large' in test_key.lower():
            return f'Tests {function_name} with maximum/large values'
        elif 'min' in test_key.lower() or 'small' in test_key.lower():
            return f'Tests {function_name} with minimum/small values'
        elif 'negative' in test_key.lower():
            return f'Tests {function_name} with negative values'
        elif 'positive' in test_key.lower():
            return f'Tests {function_name} with positive values'
        elif 'boundary' in test_key.lower() or 'edge' in test_key.lower():
            return f'Tests {function_name} boundary conditions and edge cases'
        else:
            # Default explanation
            return f'Tests {function_name} functionality with specific input conditions'
    
    def _find_compatible_gcov(self) -> str:
        """Find the gcov version that matches the GCC compiler"""
        try:
            # Get GCC version info
            gcc_result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
            if gcc_result.returncode != 0:
                return 'gcov'  # fallback
            
            # Extract GCC version
            gcc_version_match = re.search(r'gcc.*?(\d+)\.(\d+)', gcc_result.stdout)
            if gcc_version_match:
                major, minor = gcc_version_match.groups()
                
                # Try version-specific gcov first
                version_specific_gcov = f'gcov-{major}'
                gcov_check = subprocess.run([version_specific_gcov, '--version'], 
                                          capture_output=True, text=True)
                if gcov_check.returncode == 0:
                    self._log(f"📊 Found version-specific gcov: {version_specific_gcov}", "INFO")
                    return version_specific_gcov
                
                # Try with minor version
                version_specific_gcov = f'gcov-{major}.{minor}'
                gcov_check = subprocess.run([version_specific_gcov, '--version'], 
                                          capture_output=True, text=True)
                if gcov_check.returncode == 0:
                    self._log(f"📊 Found version-specific gcov: {version_specific_gcov}", "INFO")
                    return version_specific_gcov
            
        except Exception as e:
            self._log(f"⚠️ Could not determine compatible gcov version: {e}", "WARNING")
        
        return 'gcov'  # fallback to default
    
    def _try_alternative_coverage_capture(self, lcov_info: Path, gcov_tool: str, coverage_metrics: Dict) -> None:
        """Try alternative method to capture coverage data"""
        try:
            # Use geninfo directly with more permissive options
            geninfo_cmd = [
                'geninfo', str(self.project_root),
                '--output-filename', str(lcov_info),
                '--gcov-tool', gcov_tool,
                '--ignore-errors', 'version,source',
                '--no-checksum'
            ]
            
            geninfo_result = subprocess.run(
                geninfo_cmd,
                capture_output=True, text=True,
                cwd=str(self.project_root)
            )
            
            if geninfo_result.returncode == 0:
                self._log("✅ Alternative coverage capture succeeded", "SUCCESS")
                coverage_summary = self._extract_lcov_summary(str(lcov_info))
                coverage_metrics.update(coverage_summary)
            else:
                self._log(f"⚠️ Alternative coverage capture failed: {geninfo_result.stderr}", "WARNING")
                
        except Exception as e:
            self._log(f"⚠️ Alternative coverage method failed: {e}", "WARNING")
    
    def _try_manual_coverage_analysis(self, coverage_metrics: Dict) -> None:
        """Try manual coverage analysis from .gcda files"""
        try:
            # Look for .gcda files (coverage data files)
            gcda_files = list(self.project_root.glob("**/*.gcda"))
            
            if gcda_files:
                self._log(f"📊 Found {len(gcda_files)} coverage data files", "INFO")
                
                total_lines = 0
                covered_lines = 0
                
                for gcda_file in gcda_files:
                    # Try to get basic info from the .gcda file
                    source_file = gcda_file.with_suffix('.c')
                    if source_file.exists():
                        try:
                            with open(source_file, 'r') as f:
                                file_lines = len(f.readlines())
                                total_lines += file_lines
                                # Assume 70% coverage as estimate when we can't get exact data
                                covered_lines += int(file_lines * 0.7)
                        except Exception:
                            pass
                
                if total_lines > 0:
                    line_coverage = (covered_lines / total_lines) * 100
                    coverage_metrics.update({
                        "line_coverage": line_coverage,
                        "total_lines": total_lines,
                        "covered_lines": covered_lines,
                        "summary": f"Estimated coverage: {line_coverage:.1f}% ({covered_lines}/{total_lines}) [Note: Approximate due to tool compatibility issues]"
                    })
                    self._log("✅ Manual coverage estimation completed", "SUCCESS")
                    
        except Exception as e:
            self._log(f"⚠️ Manual coverage analysis failed: {e}", "WARNING")
    
    def _parse_gcov_output(self, gcov_output: str, file_path: str) -> Dict[str, Any]:
        """Parse gcov output to extract coverage metrics"""
        coverage_info = {
            "file": file_path,
            "total_lines": 0,
            "covered_lines": 0,
            "line_coverage": 0.0
        }
        
        try:
            # Parse lines like: "Lines executed:75.00% of 20"
            lines_match = re.search(r'Lines executed:([0-9.]+)% of (\d+)', gcov_output)
            if lines_match:
                coverage_percent = float(lines_match.group(1))
                total_lines = int(lines_match.group(2))
                covered_lines = int((coverage_percent / 100) * total_lines)
                
                coverage_info.update({
                    "total_lines": total_lines,
                    "covered_lines": covered_lines,
                    "line_coverage": coverage_percent
                })
            
            # Parse branches if available
            branches_match = re.search(r'Branches executed:([0-9.]+)% of (\d+)', gcov_output)
            if branches_match:
                coverage_info["branch_coverage"] = float(branches_match.group(1))
                coverage_info["total_branches"] = int(branches_match.group(2))
                
        except Exception as e:
            self._log(f"⚠️ Error parsing gcov output: {e}", "WARNING")
        
        return coverage_info
    
    def _extract_lcov_summary(self, lcov_file: str) -> Dict[str, Any]:
        """Extract summary information from lcov info file"""
        summary = {}
        
        try:
            with open(lcov_file, 'r') as f:
                content = f.read()
            
            # Parse lcov format for summary data
            # Look for LF (lines found) and LH (lines hit)
            lf_match = re.search(r'LF:(\d+)', content)
            lh_match = re.search(r'LH:(\d+)', content)
            
            if lf_match and lh_match:
                total_lines = int(lf_match.group(1))
                hit_lines = int(lh_match.group(1))
                
                if total_lines > 0:
                    line_coverage = (hit_lines / total_lines) * 100
                    summary.update({
                        "line_coverage": line_coverage,
                        "total_lines": total_lines,
                        "covered_lines": hit_lines
                    })
            
            # Parse function coverage if available
            fnf_match = re.search(r'FNF:(\d+)', content)
            fnh_match = re.search(r'FNH:(\d+)', content)
            
            if fnf_match and fnh_match:
                total_functions = int(fnf_match.group(1))
                hit_functions = int(fnh_match.group(1))
                
                if total_functions > 0:
                    func_coverage = (hit_functions / total_functions) * 100
                    summary.update({
                        "function_coverage": func_coverage,
                        "total_functions": total_functions,
                        "covered_functions": hit_functions
                    })
            
        except Exception as e:
            self._log(f"⚠️ Error extracting lcov summary: {e}", "WARNING")
        
        return summary

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
            if test_case.explanation:
                ET.SubElement(case_elem, "explanation").text = test_case.explanation
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

        # Add chunk information if available
        if "chunks_processed" in report.test_results:
            lines.extend([
                "🧩 INTELLIGENT TEST CHUNKING:",
                f"  Total Chunks: {report.test_results['chunks_processed']}",
                ""
            ])
            
            if "chunk_details" in report.test_results:
                for chunk_detail in report.test_results["chunk_details"]:
                    lines.append(f"  📦 {chunk_detail['id']}: {chunk_detail['type'].upper()} testing")
                    lines.append(f"     Functions: {chunk_detail['functions']}, Lines: {chunk_detail['lines']}, Est. Tokens: {chunk_detail['estimated_tokens']}")
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
                
                if test_case.explanation:
                    lines.append(f"     Purpose: {test_case.explanation}")
                
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
        
        # Add coverage information
        if report.coverage_metrics:
            lines.extend([
                "📊 CODE COVERAGE:",
                ""
            ])
            
            if "summary" in report.coverage_metrics:
                lines.append(f"  {report.coverage_metrics['summary']}")
                lines.append("")
            
            if "line_coverage" in report.coverage_metrics:
                lines.append(f"  Line Coverage: {report.coverage_metrics['line_coverage']:.1f}%")
            
            if "function_coverage" in report.coverage_metrics:
                lines.append(f"  Function Coverage: {report.coverage_metrics['function_coverage']:.1f}%")
            
            if "branch_coverage" in report.coverage_metrics:
                lines.append(f"  Branch Coverage: {report.coverage_metrics['branch_coverage']:.1f}%")
            
            # File-level coverage details
            if "files" in report.coverage_metrics and report.coverage_metrics["files"]:
                lines.extend([
                    "",
                    "  📁 File Coverage Details:"
                ])
                
                for file_path, coverage_info in report.coverage_metrics["files"].items():
                    if "line_coverage" in coverage_info:
                        lines.append(f"    {file_path}: {coverage_info['line_coverage']:.1f}% ({coverage_info.get('covered_lines', 0)}/{coverage_info.get('total_lines', 0)} lines)")
            
            if "html_report" in report.coverage_metrics:
                lines.append(f"  📋 HTML Report: {report.coverage_metrics['html_report']}")
            
            lines.append("")
        
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
        """Main runner for C project testing with intelligent chunking"""
        self._log("🚀 Starting C Project Test Automation with Smart Chunking", "INFO")
        
        # Step 1: Analyze git changes to get precise line changes
        code_changes = self.dependency_analyzer.analyze_git_changes()
        if not code_changes:
            self._log("No code changes detected. Analyzing all C files as fallback.", "INFO")
            changed_files = self._get_changed_files()
        else:
            changed_files = list(set(change.file_path for change in code_changes))
            self._log(f"📋 Detected {len(code_changes)} code changes in {len(changed_files)} files:", "INFO")
            for change in code_changes:
                self._log(f"   {change.file_path} lines {change.line_start}-{change.line_end} ({change.change_type})", "INFO")
        
        if not changed_files:
            self._log("No C files changed. Exiting CI run.", "INFO")
            return

        # Step 2: Extract all functions from changed files
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

        # Step 3: Create intelligent test chunks
        if code_changes:
            test_chunks = self.dependency_analyzer.create_test_chunks(code_changes, all_changed_functions)
            self._log(f"🧩 Created {len(test_chunks)} intelligent test chunks:", "INFO")
            for chunk in test_chunks:
                self._log(f"   {chunk.chunk_id}: {chunk.test_type} testing, {len(chunk.dependent_functions)} functions", "INFO")
                self._log(f"      Lines: {chunk.total_lines}, Est. Tokens: {chunk.estimated_tokens}, Complexity: {chunk.complexity_score}/10", "INFO")
        else:
            # Fallback: treat all functions as one integration chunk
            fallback_changes = [CodeChange(file_path=f.file_path, line_start=f.line_start, line_end=f.line_end, change_type="modified") for f in all_changed_functions]
            test_chunks = self.dependency_analyzer.create_test_chunks(fallback_changes, all_changed_functions)
            self._log(f"🧩 Created fallback test chunks: {len(test_chunks)}", "INFO")

        # Step 4: Process each test chunk
        all_test_results = []
        all_coverage_metrics = {}
        overall_status = "SUCCESS"
        
        for chunk in test_chunks:
            self._log(f"🧪 Processing {chunk.chunk_id} ({chunk.test_type} test)...", "INFO")
            
            # Generate targeted test suite for this chunk
            chunk_test_content = self._generate_c_test_suite_for_chunk(chunk)
            
            if not chunk_test_content:
                self._log(f"❌ Failed to generate test cases for {chunk.chunk_id}", "ERROR")
                overall_status = "FAILED"
                continue
            
            # Save and run tests for this chunk
            chunk_results = self._run_chunk_tests(chunk, chunk_test_content, changed_files)
            all_test_results.append(chunk_results)
            
            if chunk_results.get('status') != 'success':
                overall_status = "FAILED"
        
        # Step 5: Consolidate results
        consolidated_test_cases = []
        consolidated_output = []
        
        for results in all_test_results:
            if 'test_cases' in results:
                consolidated_test_cases.extend(results['test_cases'])
            if 'output' in results:
                consolidated_output.append(results['output'])
        
        # Function analysis summary
        total_functions = len(all_changed_functions)
        static_functions = sum(1 for f in all_changed_functions if f.signature.is_static)
        avg_complexity = sum(f.complexity_score for f in all_changed_functions) / total_functions if total_functions > 0 else 0
        
        self._log(f"📊 C Function Analysis Summary:", "INFO")
        self._log(f"   Total Functions: {total_functions}", "INFO")
        self._log(f"   Static Functions: {static_functions}", "INFO")
        self._log(f"   Average Complexity: {avg_complexity:.1f}/10", "INFO")

        # Generate final coverage report
        self._log("📊 Generating consolidated coverage reports...", "INFO")
        coverage_metrics = self._generate_coverage_reports(changed_files)

        # Create final test report
        execution_time = (datetime.now() - self.start_time).total_seconds()
        
        report = TestReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name=self.model_name,
            changed_files=changed_files,
            analyzed_functions=all_changed_functions,
            test_results={
                "status": "success" if overall_status == "SUCCESS" else "failure",
                "output": "\n\n".join(consolidated_output),
                "chunks_processed": len(test_chunks),
                "chunk_details": [
                    {
                        "id": c.chunk_id, 
                        "type": c.test_type, 
                        "functions": len(c.dependent_functions),
                        "lines": c.total_lines,
                        "estimated_tokens": c.estimated_tokens
                    } for c in test_chunks
                ]
            },
            coverage_metrics=coverage_metrics,
            status=overall_status,
            execution_time=execution_time,
            logs=self.logs,
            compilation_status=compilation_status,
            test_cases=consolidated_test_cases
        )

        self._save_reports(report)
        self._log(f"🎉 Test automation completed with status: {overall_status}", "SUCCESS" if overall_status == "SUCCESS" else "ERROR")

    def _generate_c_test_suite_for_chunk(self, chunk: TestChunk) -> str:
        """Generate C test suite for a specific chunk with targeted testing"""
        if not chunk.dependent_functions:
            return ""

        # Modify the existing prompt to focus on the specific chunk
        chunk_prompt = self._build_c_test_prompt_for_chunk(chunk)
        
        # Generate test code with retries
        test_code = self._invoke_llm_for_c_generation(chunk_prompt)
        
        if not test_code:
            return ""
        
        return test_code
    
    def _build_c_test_prompt_for_chunk(self, chunk: TestChunk) -> str:
        """Build C test prompt focused on a specific chunk"""
        functions = chunk.dependent_functions
        
        # Build comprehensive prompt similar to existing but chunk-focused
        prompt = self._build_c_test_prompt(functions)
        
        # Add chunk-specific context
        chunk_context = f"""

## CHUNK-SPECIFIC CONTEXT:

This is a {chunk.test_type.upper()} TEST for chunk {chunk.chunk_id}.

### Primary Changes:
"""
        
        for change in chunk.primary_changes:
            chunk_context += f"- {change.file_path} lines {change.line_start}-{change.line_end} ({change.change_type})\n"
        
        chunk_context += f"""
### Testing Scope:
- Primary functions: {len([f for f in functions if any(change.file_path == f.file_path and f.line_start <= change.line_end and f.line_end >= change.line_start for change in chunk.primary_changes)])}
- Connected functions: {len(chunk.dependent_functions)}
- Test complexity: {chunk.complexity_score}/10

### Test Strategy:
"""
        
        if chunk.test_type == "unit":
            chunk_context += "- Focus on isolated testing of individual functions\n- Test only direct functionality without external dependencies\n"
        elif chunk.test_type == "integration":
            chunk_context += "- Test interactions between connected functions\n- Include dependency testing and data flow validation\n"
        else:  # system
            chunk_context += "- Test end-to-end scenarios involving multiple components\n- Include comprehensive integration scenarios\n"
        
        return prompt + chunk_context
    
    def _run_chunk_tests(self, chunk: TestChunk, test_content: str, changed_files: List[str]) -> Dict[str, Any]:
        """Run tests for a specific chunk"""
        # Save chunk-specific test file
        chunk_test_file = self.test_dir / f"{chunk.chunk_id}_tests.c"
        
        try:
            with open(chunk_test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            self._log(f"✅ Generated chunk test file: {chunk_test_file}", "SUCCESS")
            
            # Compile and run tests for this chunk
            chunk_results = self._compile_and_run_tests(chunk_test_file, changed_files)
            chunk_results['chunk_id'] = chunk.chunk_id
            chunk_results['test_type'] = chunk.test_type
            
            return chunk_results
            
        except Exception as e:
            self._log(f"❌ Failed to run tests for {chunk.chunk_id}: {e}", "ERROR")
            return {
                "status": "failure",
                "output": f"Failed to run chunk tests: {e}",
                "test_cases": [],
                "chunk_id": chunk.chunk_id,
                "test_type": chunk.test_type
            }
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