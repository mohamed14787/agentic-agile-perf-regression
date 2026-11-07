"""Workspace discovery: introspect existing projects to detect structure.

Enables building on top of existing codebases (brownfield) instead of only
greenfield projects. Discovers packages, entities, services, controllers,
and existing files to avoid regenerating what's already there.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set


def discover_workspace_state(output_dir: str) -> Dict:
    """Scan existing workspace and extract its complete structure.

    Returns a dict with:
    - package_root: detected root package (e.g., org.springframework.samples.petclinic)
    - is_brownfield: True if existing files detected
    - existing_entities: {EntityName: {field: type, ...}}
    - existing_services: [ServiceName, ...]
    - existing_controllers: [ControllerName, ...]
    - existing_repositories: [RepositoryName, ...]
    - pom_exists: True if pom.xml found
    - build_tool: 'maven' or 'gradle'
    - spring_boot_version: extracted version or None
    - existing_files: [relative paths to all .java files]
    """
    result = {
        "package_root": None,
        "is_brownfield": False,
        "existing_entities": {},
        "existing_services": [],
        "existing_controllers": [],
        "existing_repositories": [],
        "pom_exists": False,
        "build_tool": None,
        "spring_boot_version": None,
        "existing_files": [],
        "modules": []
    }

    # 1. Find all Java files across every Maven module (multi-module reactors keep
    #    their sources one level down per module). Subsumes the single-module case.
    src_roots = _find_src_roots(output_dir)
    if not src_roots:
        return result
    java_files = [f for r in src_roots for f in _find_java_files(r)]
    if not java_files:
        return result

    packages = _extract_packages(java_files)
    if packages:
        result["package_root"] = _common_package_prefix(packages)

    # 1b. Per-module map (additive). path is the module dir relative to output_dir —
    #     the same value used as the -pl / AGILE_BUILD_MODULE build selector.
    for r in src_roots:
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(r)))
        files_here = _find_java_files(r)
        result["modules"].append({
            "name": os.path.basename(module_dir),
            "path": os.path.relpath(module_dir, output_dir),
            "package_root": _common_package_prefix(_extract_packages(files_here)),
            "file_count": len(files_here),
        })

    # 2. Parse each Java file for metadata
    for java_file in java_files:
        rel_path = os.path.relpath(java_file, output_dir)
        result["existing_files"].append(rel_path)

        try:
            metadata = _parse_java_file(java_file)

            if "@Entity" in metadata.get("annotations", []):
                result["existing_entities"][metadata["class_name"]] = metadata.get("fields", {})

            if "@Service" in metadata.get("annotations", []):
                result["existing_services"].append(metadata["class_name"])

            if "@RestController" in metadata.get("annotations", []):
                result["existing_controllers"].append(metadata["class_name"])

            if "JpaRepository" in metadata.get("source", ""):
                result["existing_repositories"].append(metadata["class_name"])

        except Exception:
            # Skip files that can't be parsed
            pass

    # 3. Detect build system
    pom_path = os.path.join(output_dir, "pom.xml")
    gradle_path = os.path.join(output_dir, "build.gradle")

    if os.path.exists(pom_path):
        result["pom_exists"] = True
        result["build_tool"] = "maven"
        result["spring_boot_version"] = _extract_spring_boot_version(pom_path)
    elif os.path.exists(gradle_path):
        result["pom_exists"] = True  # gradle also exists
        result["build_tool"] = "gradle"
        result["spring_boot_version"] = _extract_spring_boot_version_gradle(gradle_path)

    # Mark as brownfield if files exist
    result["is_brownfield"] = len(result["existing_files"]) > 0

    return result


def _find_src_roots(output_dir: str) -> List[str]:
    """Find every '<module>/src/main/java' under output_dir whose module dir holds a
    pom.xml (a real Maven module). Subsumes the single-module case (root). Prunes
    build/VCS dirs and does not descend into a source tree once found."""
    roots = []
    for cur, dirs, _ in os.walk(output_dir):
        dirs[:] = [d for d in dirs
                   if d not in ("target", "build", ".git", ".gradle", "node_modules", "__pycache__")]
        if cur.replace(os.sep, "/").endswith("src/main/java"):
            module_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur)))
            if os.path.exists(os.path.join(module_dir, "pom.xml")):
                roots.append(cur)
            dirs[:] = []  # don't descend into a source tree
    return roots


def _find_java_files(src_path: str) -> List[str]:
    """Recursively find all .java files under src_path."""
    java_files = []
    for root, dirs, files in os.walk(src_path):
        # Skip test directories and common non-source directories
        dirs[:] = [d for d in dirs if d not in ("target", ".git", "__pycache__")]

        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))

    return java_files


def _extract_packages(java_files: List[str]) -> Set[str]:
    """Extract package declarations from Java files."""
    packages = set()

    for java_file in java_files:
        try:
            with open(java_file, 'r', encoding='utf-8') as f:
                content = f.read(2000)  # Only read beginning of file

            # Look for package declaration
            match = re.search(r'^\s*package\s+([\w.]+)\s*;', content, re.MULTILINE)
            if match:
                packages.add(match.group(1))
        except Exception:
            pass

    return packages


def _common_package_prefix(packages: Set[str]) -> str:
    """Find the common package prefix across all packages."""
    if not packages:
        return None

    packages_list = sorted(list(packages))
    if len(packages_list) == 1:
        return packages_list[0]

    # Find common prefix
    first = packages_list[0].split('.')

    for other in packages_list[1:]:
        other_parts = other.split('.')

        # Keep only matching prefixes
        i = 0
        while i < len(first) and i < len(other_parts) and first[i] == other_parts[i]:
            i += 1

        first = first[:i]

    return '.'.join(first) if first else None


def _parse_java_file(java_file: str) -> Dict:
    """Parse a Java file to extract class metadata."""
    with open(java_file, 'r', encoding='utf-8') as f:
        content = f.read()

    metadata = {
        "class_name": None,
        "package": None,
        "annotations": [],
        "fields": {},
        "source": content
    }

    # Extract package
    pkg_match = re.search(r'package\s+([\w.]+)\s*;', content)
    if pkg_match:
        metadata["package"] = pkg_match.group(1)

    # Extract class name
    class_match = re.search(r'(?:public\s+)?class\s+(\w+)', content)
    if class_match:
        metadata["class_name"] = class_match.group(1)

    # Extract annotations (lines before class declaration)
    class_pos = content.find('class ')
    if class_pos > 0:
        class_section = content[:class_pos]
        annotation_matches = re.findall(r'@(\w+)', class_section)
        metadata["annotations"] = annotation_matches

    # Extract field types (for entity locking)
    field_matches = re.findall(
        r'private\s+([\w<>.]+)\s+(\w+)\s*[;=]',
        content
    )
    for field_type, field_name in field_matches:
        # Simplify generic types (List<X> -> List, Optional<X> -> Optional)
        base_type = field_type.split('<')[0].split('.')[-1]
        metadata["fields"][field_name] = base_type

    return metadata


def _extract_spring_boot_version(pom_path: str) -> str:
    """Extract Spring Boot version from pom.xml."""
    try:
        with open(pom_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for spring-boot-starter-parent version
        match = re.search(
            r'<parent>.*?<artifactId>spring-boot-starter-parent</artifactId>.*?<version>([\d.]+)</version>',
            content,
            re.DOTALL
        )
        if match:
            return match.group(1)
    except Exception:
        pass

    return None


def _extract_spring_boot_version_gradle(gradle_path: str) -> str:
    """Extract Spring Boot version from build.gradle."""
    try:
        with open(gradle_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for spring boot plugin version
        match = re.search(
            r"id\s+['\"]org\.springframework\.boot['\"]\s+version\s+['\"]([^'\"]+)['\"]",
            content
        )
        if match:
            return match.group(1)
    except Exception:
        pass

    return None


def should_skip_pom_regeneration(workspace_state: Dict) -> bool:
    """Don't regenerate pom.xml if this is a brownfield project."""
    return workspace_state.get("is_brownfield", False)


def get_existing_class_names(workspace_state: Dict) -> Set[str]:
    """Get all existing class names from file paths (more reliable than metadata parsing)."""
    existing = set()

    # Extract class names directly from file paths (most reliable)
    # File: src/main/java/org/springframework/samples/petclinic/owner/Pet.java → Pet
    for file_path in workspace_state.get("existing_files", []):
        if file_path.endswith(".java"):
            # Get filename without .java extension
            filename = file_path.split("/")[-1].replace(".java", "")
            if filename:
                existing.add(filename)

    # Also add from metadata as backup
    existing.update(workspace_state.get("existing_entities", {}).keys())
    existing.update(workspace_state.get("existing_services", []))
    existing.update(workspace_state.get("existing_controllers", []))
    # Filter out None and malformed entries
    existing = {x for x in existing if x and x not in ("for", "null")}

    return existing
