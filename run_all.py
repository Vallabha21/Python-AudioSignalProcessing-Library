# Execute each laboratory in a fresh Python process and collect results.
from pathlib import Path
import subprocess,sys,json,platform
root=Path(__file__).resolve().parent
# Capture both standard output (metrics) and standard error (diagnostics).
results=[]
for script in sorted(root.glob('[0-9][0-9]_*.py')):
    # Use this interpreter, preserving the active virtual environment.
    run=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
    results.append(dict(file=script.name,passed=run.returncode==0,
                        stdout=run.stdout,stderr=run.stderr))
    print(script.name, 'PASS' if run.returncode==0 else 'FAIL',flush=True)
# Keep a machine-readable record even if one example fails.
(root/'test_report.json').write_text(json.dumps(
    dict(python=platform.python_version(),results=results),indent=2))
# A nonzero exit code makes failures visible to terminals and automation.
if not all(r['passed'] for r in results): sys.exit(1)
