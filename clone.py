#!/usr/bin/env python3
"""
Batch clone all repositories from llm_159.txt to the new path
"""

from pathlib import Path
import subprocess
import sys

# Your new paths
CLONED_ROOT = Path("/root/satd_detection/cloned_repos").resolve()
TXT_FILE = Path("/root/satd_detection/ml_159.txt").resolve()

def main():
    # Create cloned repos directory
    CLONED_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Cloning into: {CLONED_ROOT}\n")
    
    # Read URLs from llm_159.txt
    if not TXT_FILE.exists():
        print(f"ERROR: File not found: {TXT_FILE}")
        sys.exit(1)
    
    with open(TXT_FILE, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip() and not u.strip().startswith("#")]
    
    print(f"Found {len(urls)} repositories to clone.\n")
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        repo_name = url.strip().rstrip("/").split("/")[-1].rstrip(".git")
        repo_path = CLONED_ROOT / repo_name
        
        if repo_path.exists():
            print(f"[{i}/{len(urls)}] SKIP (already exists): {repo_name}")
            successful += 1
            continue
        
        print(f"[{i}/{len(urls)}] Cloning: {repo_name}...", end=" ", flush=True)
        
        try:
            result = subprocess.run(
                ["git", "clone", url, str(repo_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=300,
                check=False
            )
            
            if result.returncode == 0:
                print("✓ OK")
                successful += 1
            else:
                error_msg = result.stderr.decode('utf-8', errors='ignore')[:100]
                print(f"✗ FAILED ({error_msg})")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print("✗ TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Cloning complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(urls)}")
    print(f"\nRepositories are in: {CLONED_ROOT}")

if __name__ == "__main__":
    main()