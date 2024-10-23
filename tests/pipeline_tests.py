import pytest
import subprocess
import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def run_script(script_path, args=[]):
    command = [sys.executable, script_path] + args
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"\nOutput from {script_path}:")
    print(result.stdout)
    if result.stderr:
        print(f"Output from {script_path}:")
        print(result.stderr)
    return result

def test_fitter_3d_optimise(capsys):
    script_path = os.path.join(parent_dir, 'fitter_3d', 'optimise.py')
    result = run_script(script_path, ['--mesh_dir', 'fitter_3d/ATTA_BOI', '--scheme', 'default', '--lr', '1e-3', '--nits', '10'])
    assert result.returncode == 0, f"fitter_3d/optimise.py failed with error:\n{result.stderr}"
    
    # Capture and print the output
    captured = capsys.readouterr()
    print(captured.out)

def test_smal_fitter_optimize_to_joints(capsys):
    script_path = os.path.join(parent_dir, 'smal_fitter', 'optimize_to_joints.py')
    result = run_script(script_path, ['--test'])
    assert result.returncode == 0, f"smal_fitter/optimize_to_joints.py failed with error:\n{result.stderr}"
    
    # Capture and print the output
    captured = capsys.readouterr()
    print(captured.out)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
