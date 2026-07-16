import ast
import os

def extract_docstrings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            node = ast.parse(f.read(), filename=file_path)
        except Exception as e:
            # Log syntax errors gracefully without crashing
            return f"## File: {file_path}\n*⚠️ Skipped parsing due to Syntax Error: {e}*\n\n"

    output = [f"# File: {file_path}\n"]
    
    # Grab Module level docstring
    module_doc = ast.get_docstring(node)
    if module_doc:
        output.append(f"### Module Overview\n```text\n{module_doc}\n```\n")

    # Walk through classes and functions
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(child)
            if docstring:
                type_name = "Class" if isinstance(child, ast.ClassDef) else "Function"
                output.append(f"### {type_name}: `{child.name}`\n```text\n{docstring}\n```\n")
                
    return "".join(output) if len(output) > 1 else ""

def main():
    markdown_content = ["# Repository Docstrings\n\nGenerated via static analysis.\n\n---\n\n"]
    
    for root, dirs, files in os.walk("."):
        # Explicitly ignore hidden/build/env folders safely
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['docs', 'venv', 'env', '__pycache__']]
            
        for file in files:
            if file.endswith(".py") and file != "extract_docs.py":
                full_path = os.path.join(root, file)
                file_docs = extract_docstrings(full_path)
                if file_docs:
                    markdown_content.append(file_docs + "\n---\n\n")
                    
    with open("extracted_docstrings.md", "w", encoding="utf-8") as f:
        f.writelines(markdown_content)
        
    print("✨ Success! All available docstrings compiled into 'extracted_docstrings.md'")

if __name__ == "__main__":
    main()