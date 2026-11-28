"""The Kaggle backend, checked without touching the network.

Kaggle is the only compute the bot has on the deployment, and every way
it could go wrong ended up as a message the user could not act on: a
package that was not installed, a file that was downloaded and never
sent, a dataset name that was not a ref, a kernel that sat there until
it was killed.

So the parts that decide those things are checked here:

    python tests/test_kaggle.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT  # noqa: E402

sys.path.insert(0, ROOT)

import kaggle_backend as kb  # noqa: E402

FAILED = []
PASSED = 0


def check(label, got, want):
    global PASSED
    if got == want:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")


def check_true(label, cond):
    check(label, bool(cond), True)


# ---------------------------------------------------------------------------
# The pip preamble
# ---------------------------------------------------------------------------
print("\n-- pip install --")

check("_pip_packages finds a plain install line",
      kb._pip_packages("pip install cowsay"), [["cowsay"]])
check("_pip_packages finds a python -m install line",
      kb._pip_packages("python -m pip install numpy pandas"),
      [["numpy", "pandas"]])
check("_pip_packages finds a subprocess install",
      kb._pip_packages(
          "subprocess.run([sys.executable, '-m', 'pip', 'install', 'cowsay'])"),
      [["cowsay"]])
check("_pip_packages finds a notebook install",
      kb._pip_packages("!pip install cowsay"), [["cowsay"]])
check("_pip_packages leaves a script alone",
      kb._pip_packages("import cowsay\nprint('hi')"), [])
check("_pip_packages keeps the flags out of the name",
      kb._pip_packages("pip install --quiet cowsay"),
      [["--quiet", "cowsay"]])

# The same package twice is one install.
check("_pip_packages does not install twice",
      kb._pip_packages("pip install cowsay\npip install cowsay"),
      [["cowsay"]])

preamble, specs = kb._pip_preamble("import cowsay\n"
                                   "subprocess.run([sys.executable, '-m', 'pip', "
                                   "'install', 'cowsay'])")
check_true("_pip_preamble writes a preamble", "pip install" in preamble)
check_true("_pip_preamble installs before the script runs",
           preamble.index("pip install") < len(preamble))
check("_pip_preamble lists the package", specs, [["cowsay"]])

# The extra packages can come as names or as argument lists - one
# "cowsay" and not five letters.
check("_pip_preamble takes extra names",
      kb._pip_preamble("import x", ["cowsay"])[1], [["cowsay"]])
check("_pip_preamble takes extra argument lists",
      kb._pip_preamble("import x", [["cowsay"]])[1], [["cowsay"]])
check("_pip_preamble does not install the same thing twice",
      kb._pip_preamble("pip install cowsay", ["cowsay"])[1], [["cowsay"]])

check("_pip_preamble is empty for a script that installs nothing",
      kb._pip_preamble("print('hi')"), ("", []))

# ---------------------------------------------------------------------------
# The missing-package retry
# ---------------------------------------------------------------------------
print("\n-- the missing package retry --")

check("_missing_modules finds the name",
      kb._missing_modules("ModuleNotFoundError: No module named 'cowsay'"),
      ["cowsay"])
check("_missing_modules handles the double quotes",
      kb._missing_modules('ModuleNotFoundError: No module named "cowsay"'),
      ["cowsay"])
check("_missing_modules finds nothing in a normal traceback",
      kb._missing_modules("ValueError: this is the bug"), [])
check("_missing_modules does not repeat itself",
      kb._missing_modules(
          "ModuleNotFoundError: No module named 'cowsay'\n"
          "ModuleNotFoundError: No module named 'cowsay'"),
      ["cowsay"])

check("_packages_for maps sklearn to scikit-learn",
      kb._packages_for(["sklearn"]), ["scikit-learn"])
check("_packages_for maps cv2 to opencv-python",
      kb._packages_for(["cv2"]), ["opencv-python"])
check("_packages_for maps PIL to pillow",
      kb._packages_for(["PIL"]), ["pillow"])
check("_packages_for takes the first part of a dotted name",
      kb._packages_for(["google.protobuf"]), ["protobuf"])
check("_packages_for leaves a normal name alone",
      kb._packages_for(["cowsay"]), ["cowsay"])
check("_packages_for skips the standard library",
      kb._packages_for(["os", "json", "sys"]), [])
check("_packages_for skips the private modules",
      kb._packages_for(["_numpy"]), [])
check("_packages_for does not repeat itself",
      kb._packages_for(["cowsay", "cowsay"]), ["cowsay"])

# ---------------------------------------------------------------------------
# Durations and the quota
# ---------------------------------------------------------------------------
print("\n-- the quota --")

check("_seconds reads seconds", kb._seconds("3600s"), 3600)
check("_seconds reads hours and minutes", kb._seconds("1h 2m 3s"), 3723)
check("_seconds reads a plain number", kb._seconds(90), 90)
check("_seconds reads nothing as zero", kb._seconds(None), 0)
check("_seconds reads an empty string as zero", kb._seconds(""), 0)

# The quota itself needs the network, so the decision is checked with a
# fake one: a quota that is used up has to be refused before the run.
class _FakeQuota:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


class _FakeClient:
    def __init__(self, data):
        self._data = data

    def quota_view(self):
        return _FakeQuota(self._data)


USED_UP = {
    "gpuQuota": {"timeUsed": "21600s", "totalTimeAllowed": "21600s"},
    "tpuQuota": {"timeUsed": "0s", "totalTimeAllowed": "72000s"},
}
PLENTY = {
    "gpuQuota": {"timeUsed": "0s", "totalTimeAllowed": "21600s"},
    "tpuQuota": {"timeUsed": "0s", "totalTimeAllowed": "72000s"},
}

_real_api = kb._kaggle_api

kb._kaggle_api = lambda: _FakeClient(USED_UP)
try:
    check_true("_quota_problem refuses a GPU with no quota left",
               "quota is used up" in kb._quota_problem("NvidiaTeslaT4"))
    check("_quota_problem leaves the CPU alone",
          kb._quota_problem(""), "")
    check("_quota_problem leaves the TPU alone",
          kb._quota_problem("TpuV5E8"), "")
finally:
    kb._kaggle_api = _real_api

kb._kaggle_api = lambda: _FakeClient(PLENTY)
try:
    check("_quota_problem allows a GPU with quota left",
          kb._quota_problem("NvidiaTeslaT4"), "")
finally:
    kb._kaggle_api = _real_api

# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
print("\n-- the network --")

check_true("_looks_like_no_network sees a name that did not resolve",
           kb._looks_like_no_network(
               "socket.gaierror: [Errno -2] Name or service not known"))
check_true("_looks_like_no_network sees a refused connection",
           kb._looks_like_no_network("Connection refused"))
check("_looks_like_no_network leaves a normal error alone",
      kb._looks_like_no_network("ValueError: this is the bug"), False)

# ---------------------------------------------------------------------------
# The files
# ---------------------------------------------------------------------------
print("\n-- the files --")

state = tempfile.mkdtemp(prefix="kaggle-test-")
try:
    source = os.path.join(state, "source.csv")
    with open(source, "w") as fh:
        fh.write("a,b\n1,2\n")
    files = [{"name": "source.csv", "path": source, "size": 8},
             {"name": "run.log", "path": source, "size": 8}]

    saved = kb._save_outputs(files, os.path.join(state, "kept"))
    check("_save_outputs keeps the output file", saved, ["source.csv"])
    check_true("_save_outputs leaves the path of the copy",
               files[0].get("saved_path"))
    check_true("_save_outputs does not keep the log",
               not os.path.exists(os.path.join(state, "kept", "run.log")))
    check_true("_save_outputs keeps the file where it can be read",
               os.path.exists(files[0]["saved_path"]))

    check("_save_outputs makes the directory if it is not there",
          kb._save_outputs(files, os.path.join(state, "made", "up")),
          ["source.csv"])
    check("_save_outputs survives no directory at all",
          kb._save_outputs(files, ""), [])

    # The preamble that puts the last run's files back.
    preamble, carried = kb._carry_preamble(os.path.join(state, "kept"))
    check("_carry_preamble carries the file", carried, ["source.csv"])
    check_true("_carry_preamble writes the file back",
               "source.csv" in preamble)
    check("_carry_preamble has nothing to carry from an empty directory",
          kb._carry_preamble(os.path.join(state, "empty"))[1], [])
    check("_carry_preamble has nothing to carry from nothing",
          kb._carry_preamble("")[1], [])
finally:
    shutil.rmtree(state, ignore_errors=True)

# ---------------------------------------------------------------------------
# What the model is told
# ---------------------------------------------------------------------------
print("\n-- what the model is told --")

text = kb.format_output({
    "stdout": "hello",
    "stderr": "Traceback (most recent call last):\nValueError: nope",
    "files": [{"name": "out.csv", "path": "/tmp/out.csv", "size": 10}],
    "saved": ["out.csv"],
    "state": "ERROR",
    "success": False,
}, state_rel="kaggle/default")
check_true("the output has the stdout", "hello" in text)
check_true("the output has the traceback", "ValueError: nope" in text)
check_true("the output names the file", "out.csv" in text)
check_true("the output says where the file is",
           "read_file" in text)

text = kb.format_output({"error": "the quota is used up"})
check_true("an error is passed straight through", "quota is used up" in text)

text = kb.format_output({"stdout": "", "state": "COMPLETE", "success": True})
check_true("a run with no output says so", "no output" in text)

text = kb.format_output({
    "stdout": "ok",
    "state": "COMPLETE",
    "success": True,
    "installed": ["cowsay"],
    "extra_install_note": "the script was missing cowsay",
})
check_true("the output says what was installed", "cowsay" in text)
check_true("the output says what happened", "was missing cowsay" in text)

text = kb.format_output({
    "stdout": "",
    "state": "TIMEOUT",
    "success": False,
    "timed_out": True,
    "elapsed": 600,
})
check_true("a timeout says so", "was still running" in text)

# ---------------------------------------------------------------------------
# The datasets
# ---------------------------------------------------------------------------
print("\n-- the datasets --")


class _FakeDataset:
    def __init__(self, ref, title):
        self.ref = ref
        self.title = title


class _FakeList:
    def __init__(self, items):
        self.datasets = items


class _FakeDatasetClient:
    def __init__(self, items):
        self._items = items

    def dataset_list(self, search=None, **kwargs):
        return _FakeList(self._items)


DATASETS = [_FakeDataset("heptapod/titanic", "Titanic"),
            _FakeDataset("someone/other", "Something Else")]

_real_api = kb._kaggle_api
kb._kaggle_api = lambda: _FakeDatasetClient(DATASETS)
try:
    check("_find_dataset finds a dataset by its slug",
          kb._find_dataset("titanic"), "heptapod/titanic")
    check("_find_dataset finds a dataset by its title",
          kb._find_dataset("Titanic"), "heptapod/titanic")
    check("_find_dataset finds nothing that is not there",
          kb._find_dataset("nothing-like-this"), "")

    refs, notes = kb._resolve_datasets(["titanic"])
    check("_resolve_datasets turns a bare name into a ref",
          refs, ["heptapod/titanic"])
    check_true("_resolve_datasets says what it did", notes)

    refs, notes = kb._resolve_datasets(["owner/ref"])
    check("_resolve_datasets leaves a full ref alone", refs, ["owner/ref"])

    refs, notes = kb._resolve_datasets(["nothing-like-this"])
    check("_resolve_datasets drops a name it cannot find", refs, [])
    check_true("_resolve_datasets says what it could not find", notes)
finally:
    kb._kaggle_api = _real_api

# ---------------------------------------------------------------------------
# The credentials
# ---------------------------------------------------------------------------
print("\n-- the credentials --")


class _Refusal(Exception):
    status_code = 401


check_true("_is_auth_refusal sees a 401",
           kb._is_auth_refusal(_Refusal("401 Client Error")))
check_true("_is_auth_refusal sees the word",
           kb._is_auth_refusal(Exception("Unauthenticated")))
check("_is_auth_refusal leaves other errors alone",
      kb._is_auth_refusal(Exception("500 Server Error")), False)
check("_is_auth_refusal leaves a normal error alone",
      kb._is_auth_refusal(Exception("ValueError: nope")), False)

# The refusal has to arrive as something a person can act on.
check_true("the refusal says the token has to be re-issued",
           "re-issued" in kb.AUTH_REFUSED)
check_true("the refusal says the script is not at fault",
           "Nothing is wrong with the script" in kb.AUTH_REFUSED)


class _RefusingClient:
    def __init__(self, refuse=True):
        self._refuse = refuse

    def dataset_list(self, search=None, **kwargs):
        return [_FakeDataset("heptapod/titanic", "Titanic")]

    def quota_view(self):
        if self._refuse:
            raise _Refusal("401 Client Error: Unauthorized")
        return _FakeQuota(PLENTY)


_real_api = kb._kaggle_api

kb._kaggle_api = lambda: _RefusingClient(refuse=True)
try:
    # The credential can read datasets, so it is accepted - but the
    # kernels service refuses it, and that has to be said plainly.
    report = kb.kaggle_status()
    check("the credential is accepted", report["authenticated"], True)
    check("the kernels are refused", report["kernels_allowed"], False)
    check_true("the report says why",
               "re-issued" in (report["error"] or ""))
    check_true("the prompt hint says the token has to be re-issued",
               "re-issued" in kb._hint_from_status(report))
    check_true("the quota check refuses too",
               "re-issued" in kb._quota_problem(""))
    check_true("the quota call reports the refusal",
               "re-issued" in (kb._quota_sync().get("error") or ""))
finally:
    kb._kaggle_api = _real_api

kb._kaggle_api = lambda: _RefusingClient(refuse=False)
try:
    report = kb.kaggle_status()
    check("the credential is accepted", report["authenticated"], True)
    check("the kernels are allowed", report["kernels_allowed"], True)
    check_true("the prompt hint says it works",
               "working" in kb._hint_from_status(report))
finally:
    kb._kaggle_api = _real_api

# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
print("\n-- the result --")

result = asyncio.run(kb.run_code("", timeout=10))
check("a run with no code says so", result.get("error"), "no code given")

print()
print("=" * 60)
if FAILED:
    print(f"{len(FAILED)} of {PASSED + len(FAILED)} failed:")
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)
print(f"all {PASSED} checks passed")
