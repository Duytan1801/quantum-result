"""Run every test in this directory.

    python tests/run_all.py

The quick ones run first. The ones that talk to the model are marked
LIVE, and they are the slow ones - the point of the suite is that the
cheap checks run in a few seconds and the live ones are run when there
is time.

    python tests/run_all.py          # everything
    python tests/run_all.py quick    # only the ones that do not use the network
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# The bot's own file tools work out where to write from the working
# directory, so every test runs in a scratch directory of its own. A test
# run then leaves nothing in the repository.
SCRATCH = tempfile.mkdtemp(prefix="bot-tests-")

# (file, needs the network)
QUICK = [
    "regress.py",
    "test_guard.py",
    "test_cache.py",
    "test_handler.py",
    "test_addresses.py",
    "test_detached.py",
    "test_newsfeed.py",
    "test_retry.py",
    "test_proactive.py",
    "test_model_command.py",
    "test_kaggle.py",
]

LIVE = [
    "test_decisions.py",
    "live_probe_super.py",
    "probe_tool_retry.py",
    "probe_personal_deep.py",
    "probe_natural.py",
    "test_detached_store.py",
]


def run(name):
    print()
    print("#" * 70)
    print(f"# {name}")
    print("#" * 70)
    scratch = os.path.join(SCRATCH, name.replace(".py", ""))
    os.makedirs(scratch, exist_ok=True)
    result = subprocess.run([sys.executable, os.path.join(HERE, name)],
                            cwd=scratch)
    return result.returncode == 0


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    if mode == "quick":
        files = QUICK
    elif mode == "live":
        files = LIVE
    else:
        files = QUICK + LIVE

    failed = []
    for name in files:
        if not run(name):
            failed.append(name)

    print()
    print("=" * 70)
    if failed:
        print(f"{len(failed)} of {len(files)} failed: {', '.join(failed)}")
        shutil.rmtree(SCRATCH, ignore_errors=True)
        sys.exit(1)
    print(f"all {len(files)} passed")
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
