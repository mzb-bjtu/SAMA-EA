"""Convert native RCMP benchmark files to structured JSON."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBS = ["RCMPSPLIB", "MPLIB1", "MPLIB2"]

def load_converter():
    spec = importlib.util.spec_from_file_location(
        "rcmp2json", HERE / "RCMP2JSON.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RCMPAutoConverter

def main():
    Conv = load_converter()
    ok_total, fail_total = 0, 0
    for lib in LIBS:
        in_dir = HERE / "RCMP" / lib
        out_dir = HERE / "JSON" / lib
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("*.rcmp"))
        print(f"== {lib}: {len(files)} files -> {out_dir.relative_to(HERE)} ==")
        for f in files:
            ok, msg = Conv.convert_file(str(f), str(out_dir))
            print(("  OK   " if ok else "  FAIL ") + msg)
            ok_total += ok
            fail_total += (not ok)
    print(f"\ndone: {ok_total} converted, {fail_total} failed")
    raise SystemExit(1 if fail_total else 0)

if __name__ == "__main__":
    main()
