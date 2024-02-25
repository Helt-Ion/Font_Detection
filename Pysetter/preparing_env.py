import re
import subprocess
import os
import shutil
import sys
import importlib.util
import platform

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)
python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
dir_repos = "repositories"


def check_python_version():
    is_windows = platform.system() == "Windows"
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if is_windows:
        supported_minors = [10]
    else:
        supported_minors = [7, 8, 9, 10, 11]

    if not (major == 3 and minor in supported_minors):
        print(f"""
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.6 Python, but you have {major}.{minor}.{micro}.
If you encounter an error with "RuntimeError: Couldn't install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in WebUI's directory.

You can download 3.10 Python from here: https://www.python.org/downloads/release/python-3106/
""")


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = False) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None):
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}',
               desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    return result.returncode == 0


def git_fix_workspace(dir, name):
    run(f'"{git}" -C "{dir}" fetch --refetch --no-auto-gc',
        f"Fetching all contents for {name}", f"Couldn't fetch {name}", live=True)
    run(f'"{git}" -C "{dir}" gc --aggressive --prune=now', f"Pruning {name}", f"Couldn't prune {name}", live=True)
    return


def run_git(dir, name, command, desc=None, errdesc=None, custom_env=None, live: bool = False, autofix=True):
    try:
        return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)
    except RuntimeError:
        if not autofix:
            raise

    print(f"{errdesc}, attempting autofix...")
    git_fix_workspace(dir, name)

    return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)


def git_clone(url, dir, name, commithash=None):
    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run_git(dir, name, 'rev-parse HEAD', None,
                               f"Couldn't determine {name}'s hash: {commithash}", live=False).strip()
        if current_hash == commithash:
            return

        if run_git(dir, name, 'config --get remote.origin.url', None,
                   f"Couldn't determine {name}'s origin URL", live=False).strip() != url:
            run_git(dir, name, f'remote set-url origin "{url}"', None,
                    f"Failed to set {name}'s origin URL", live=False)

        run_git(dir, name, 'fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}", autofix=False)

        run_git(dir, name, f'checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...",
                f"Couldn't checkout commit {commithash} for {name}", live=True)

        return

    try:
        run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}", live=True)
    except RuntimeError:
        shutil.rmtree(dir, ignore_errors=True)
        raise

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def prepare_environment():
    print("Preparing Environment...")

    install_torch = os.environ.get('INSTALL_TORCH', "True")
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_package = os.environ.get('TORCH_PACKAGE', "torch==2.1.2")
    torchvision_package = os.environ.get('TORCHVISION_PACKAGE', "torchvision==0.16.2")
    torch_command = f"install {torch_package} {torchvision_package} --no-cache-dir --no-warn-script-location --index-url {torch_index_url}"

    install_xformers = os.environ.get('INSTALL_XFORMERS', "True")
    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers')
    xformers_command = f"install -U -I {xformers_package} --no-deps --no-warn-script-location --index-url {torch_index_url}"

    install_accelerate = os.environ.get('INSTALL_ACCELERATE', "True")
    accelerate_package = os.environ.get('ACCELERATE_PACKAGE', 'accelerate-0.22.0-py3-none-any.whl')
    accelerate_command = f"install {accelerate_package} --no-warn-script-location"

    requirements_file = os.environ.get('REQS_FILE', "requirements.txt")
    
    check_python_version()
    print(f"Using Python {sys.version}")
    
    if install_torch == "True":
        if not is_installed("torch") or not is_installed("torchvision"):
            run(f'"{python}" -m pip {torch_command}', "Installing torch and torchvision",
                "Couldn't install torch", live=True)

    if install_xformers == "True":
        if not is_installed("xformers"):
            run_pip(xformers_command, "xformers")

    if install_accelerate == "True":
        if not is_installed("accelerate"):
            run_pip(accelerate_command, "accelerate")

    run_pip(f"install -r \"{requirements_file}\" --no-warn-script-location", "requirements")

    print("Preparing Environment Finished!")


def update_environment():
    accelerate_package = os.environ.get('ACCELERATE_PACKAGE', 'accelerate-0.22.0-py3-none-any.whl')
    accelerate_command = f"install {accelerate_package} --force-reinstall --no-deps --no-warn-script-location"
    if is_installed("accelerate"):
        run(f'"{python}" -m pip {accelerate_command}', "Updating Environment...")
