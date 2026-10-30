"""Offline smoke test of the whole orchestrator on the scripted model. From tests/:
  PYTHONPATH=..:. python _run_orchestrator.py --models-file test_models.yaml --judge scripted/judge --out /tmp/vdb --rounds 2 --chains 2 --vb-epochs 2 --questionnaires LTO --scenarios-limit 3 --checkpoints 0 2 --display none"""
import runpy, sys
import scripted_model  # noqa: F401
sys.argv = ["run_all.py"] + sys.argv[1:]
runpy.run_path(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"), run_name="__main__")
