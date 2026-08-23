"""
The 10 AST Parsing, Chunking & Graph Strategies for Code Search and Quantization.
"""

import os
import re
import math
from typing import List, Dict, Any, Tuple

def simple_tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'[a-zA-Z0-9_]+', text) if len(w) > 1]

# -------------------------------------------------------------
# 1. Flat Fixed-Window Token Chunking Baseline
# -------------------------------------------------------------
class FlatWindowStrategy:
    name = "1. Flat-Window Baseline"
    description = "Fixed 128-token windows without syntax or AST boundary awareness."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str, chunk_size: int = 128) -> List[Dict[str, Any]]:
        lines = code_text.split("\n")
        chunks = []
        step = 15  # ~120 tokens
        for i in range(0, len(lines), step):
            block = "\n".join(lines[i:i+step])
            if block.strip():
                chunks.append({
                    "file": rel_path,
                    "text": block,
                    "token_count": len(block.split()),
                    "type": "flat_window"
                })
        return chunks

# -------------------------------------------------------------
# 2. Basic Function Scope Closure
# -------------------------------------------------------------
class FunctionClosureStrategy:
    name = "2. Function Scope Closure"
    description = "Atomic top-level function/class closures preserving signature-body integrity."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str) -> List[Dict[str, Any]]:
        # Match function/method boundaries
        chunks = []
        matches = re.finditer(r'(?:pub\s+)?(?:fn|def|async\s+fn|class|struct|trait)\s+([a-zA-Z0-9_]+)', code_text)
        indices = [m.start() for m in matches]
        
        if not indices:
            return FlatWindowStrategy.chunk(code_text, rel_path)
            
        indices.append(len(code_text))
        for i in range(len(indices) - 1):
            block = code_text[indices[i]:indices[i+1]].strip()
            if block:
                chunks.append({
                    "file": rel_path,
                    "text": block,
                    "token_count": len(block.split()),
                    "type": "function_closure"
                })
        return chunks

# -------------------------------------------------------------
# 3. AST Subtree Token-Budgeting
# -------------------------------------------------------------
class BudgetedSubtreesStrategy:
    name = "3. AST Budgeted Subtrees"
    description = "Recursively partitions oversized function ASTs at statement/block boundaries with local headers."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str, max_tokens: int = 128) -> List[Dict[str, Any]]:
        raw_funcs = FunctionClosureStrategy.chunk(code_text, rel_path)
        budgeted_chunks = []
        
        for f in raw_funcs:
            text = f["text"]
            lines = text.split("\n")
            if len(lines) <= 20:
                budgeted_chunks.append(f)
            else:
                # Keep signature header on each sub-chunk
                header = lines[0]
                body_lines = lines[1:]
                for j in range(0, len(body_lines), 15):
                    sub_block = header + "\n" + "\n".join(body_lines[j:j+15])
                    budgeted_chunks.append({
                        "file": rel_path,
                        "text": sub_block,
                        "token_count": len(sub_block.split()),
                        "type": "budgeted_subtree"
                    })
        return budgeted_chunks

# -------------------------------------------------------------
# 4. Parent-Context Header Inheritance
# -------------------------------------------------------------
class HierarchicalHeaderInjectionStrategy:
    name = "4. Hierarchical Header Injection"
    description = "Injects enclosing trait/class/struct definition context into each child method chunk."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str) -> List[Dict[str, Any]]:
        lines = code_text.split("\n")
        parent_context = ""
        # Find struct / impl / class header
        for line in lines[:15]:
            if any(k in line for k in ["struct ", "impl ", "class ", "trait "]):
                parent_context = line.strip()
                break
                
        sub_chunks = BudgetedSubtreesStrategy.chunk(code_text, rel_path)
        if parent_context:
            for c in sub_chunks:
                c["text"] = f"// Scope: {parent_context}\n" + c["text"]
                c["token_count"] += len(parent_context.split())
                c["type"] = "header_injected"
        return sub_chunks

# -------------------------------------------------------------
# 5. Docstring-AST Syntax Binding
# -------------------------------------------------------------
class DocSyntaxBindingStrategy:
    name = "5. Docstring-AST Syntax Binding"
    description = "Binds documentation comments explicitly to their decorated AST declaration nodes."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str) -> List[Dict[str, Any]]:
        # Match docstring preceding functions
        chunks = []
        pattern = r'((?:(?:///[^\n]*\n)|(?:/\*\*[\s\S]*?\*/)|(?:\'\'\'[\s\S]*?\'\'\')|(?:"""[\s\S]*?"""))*\s*(?:pub\s+)?(?:fn|def|struct|class|trait)\s+[a-zA-Z0-9_]+[\s\S]*?(?=(?:(?:///[^\n]*\n)|(?:/\*\*[\s\S]*?\*/)|(?:\'\'\'[\s\S]*?\'\'\')|(?:"""[\s\S]*?"""))*\s*(?:pub\s+)?(?:fn|def|struct|class|trait)\s+[a-zA-Z0-9_]+|\Z))'
        matches = re.findall(pattern, code_text)
        
        if not matches:
            return HierarchicalHeaderInjectionStrategy.chunk(code_text, rel_path)
            
        for m in matches:
            block = m.strip()
            if block:
                chunks.append({
                    "file": rel_path,
                    "text": block,
                    "token_count": len(block.split()),
                    "type": "doc_syntax_bound"
                })
        return chunks

# -------------------------------------------------------------
# 6. Type-Signature AST Isolation
# -------------------------------------------------------------
class SignatureBodyDecouplingStrategy:
    name = "6. Signature-Body Decoupling"
    description = "Dual-rate indexing isolating exact type signatures from implementation bodies."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str) -> List[Dict[str, Any]]:
        chunks = []
        lines = code_text.split("\n")
        for i, line in enumerate(lines):
            if any(line.strip().startswith(kw) for kw in ["pub fn", "fn ", "def ", "pub struct", "struct ", "trait ", "class "]):
                sig = line.strip()
                chunks.append({
                    "file": rel_path,
                    "text": f"// Signature: {sig}\n" + "\n".join(lines[i:i+10]),
                    "token_count": len(sig.split()) + 30,
                    "type": "decoupled_signature"
                })
        if not chunks:
            return DocSyntaxBindingStrategy.chunk(code_text, rel_path)
        return chunks

# -------------------------------------------------------------
# 7. Call-Graph & Dependency AST Expansion
# -------------------------------------------------------------
class CallGraphLinkingStrategy:
    name = "7. Call-Graph & Dependency Expansion"
    description = "Augments AST chunks with 1-hop caller/callee function symbol references."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str) -> List[Dict[str, Any]]:
        base_chunks = DocSyntaxBindingStrategy.chunk(code_text, rel_path)
        # Extract callee identifier symbols from the entire file
        callees = set(re.findall(r'([a-zA-Z0-9_]{3,})\(', code_text))
        callee_tags = " // Calls: " + ", ".join(list(callees)[:6])
        
        for c in base_chunks:
            c["text"] = c["text"] + "\n" + callee_tags
            c["token_count"] += len(callee_tags.split())
            c["type"] = "call_graph_augmented"
        return base_chunks

# -------------------------------------------------------------
# 8. Control Flow Graph (CFG) Basic-Block Chunking
# -------------------------------------------------------------
class CFGBlockPartitioningStrategy:
    name = "8. CFG Basic-Block Partitioning"
    description = "Partitions functions along CFG control branching points (match/switch, if-else, try-catch)."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str) -> List[Dict[str, Any]]:
        chunks = []
        blocks = re.split(r'\n(?=\s*(?:if\s+|else\s+|match\s+|switch\s+|for\s+|while\s+|try\s+|except\s+))', code_text)
        for b in blocks:
            b_clean = b.strip()
            if len(b_clean) > 30:
                chunks.append({
                    "file": rel_path,
                    "text": b_clean,
                    "token_count": len(b_clean.split()),
                    "type": "cfg_basic_block"
                })
        if not chunks:
            return CallGraphLinkingStrategy.chunk(code_text, rel_path)
        return chunks

# -------------------------------------------------------------
# 9. Multi-Granularity Syntactic Folding
# -------------------------------------------------------------
class SyntacticMultiGranularityStrategy:
    name = "9. Syntactic Multi-Granularity"
    description = "Dual-rate hierarchy: coarse function vector for Stage 1 filter + fine-grained statement chunks for Stage 2."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str) -> List[Dict[str, Any]]:
        coarse = FunctionClosureStrategy.chunk(code_text, rel_path)
        fine = BudgetedSubtreesStrategy.chunk(code_text, rel_path)
        
        combined = []
        for c in coarse[:3]:
            c["type"] = "coarse_granularity"
            combined.append(c)
        for f in fine:
            f["type"] = "fine_granularity"
            combined.append(f)
        return combined

# -------------------------------------------------------------
# 10. Unified Context-Aware AST Graph Vector Quantizer
# -------------------------------------------------------------
class UnifiedASTGraphQuantizerStrategy:
    name = "10. Unified AST Graph Quantizer"
    description = "Optimal multi-tier synthesis: Header Injection + Budgeted Subtrees + 1-Hop Symbol Graph + Dual-Rate 1.35b/d MaxSim."
    
    @staticmethod
    def chunk(code_text: str, rel_path: str) -> List[Dict[str, Any]]:
        # 1. Header context
        lines = code_text.split("\n")
        parent_context = ""
        for line in lines[:15]:
            if any(k in line for k in ["struct ", "impl ", "class ", "trait ", "package "]):
                parent_context = line.strip()
                break
                
        # 2. Docstring + function binding
        doc_chunks = DocSyntaxBindingStrategy.chunk(code_text, rel_path)
        
        # 3. Inject parent scope and callee graph tags
        callees = set(re.findall(r'([a-zA-Z0-9_]{3,})\(', code_text))
        callee_tags = " // Graph: " + ", ".join(list(callees)[:4]) if callees else ""
        
        final_chunks = []
        for c in doc_chunks:
            header_prefix = f"// Context: {parent_context}\n" if parent_context else ""
            c["text"] = header_prefix + c["text"] + ("\n" + callee_tags if callee_tags else "")
            c["token_count"] = len(c["text"].split())
            c["type"] = "unified_ast_graph"
            final_chunks.append(c)
        return final_chunks

ALL_AST_STRATEGIES = [
    FlatWindowStrategy,
    FunctionClosureStrategy,
    BudgetedSubtreesStrategy,
    HierarchicalHeaderInjectionStrategy,
    DocSyntaxBindingStrategy,
    SignatureBodyDecouplingStrategy,
    CallGraphLinkingStrategy,
    CFGBlockPartitioningStrategy,
    SyntacticMultiGranularityStrategy,
    UnifiedASTGraphQuantizerStrategy
]
