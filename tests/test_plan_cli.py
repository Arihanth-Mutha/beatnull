import json

import pytest

import beatnull as bn
from beatnull import demo
from beatnull.cli import main
from beatnull.plan import Plan, PlanError, load_plan, save_plan


@pytest.fixture
def noise_csv(tmp_path):
    path = tmp_path / "noise.csv"
    demo.write_csv(demo.simulate("noise"), path)
    return path


def test_plan_round_trip_is_intact(tmp_path):
    path = tmp_path / "plan.json"
    digest = save_plan(Plan(metric="hit", controls=("zero",), group_column="suite"), path)
    status = load_plan(path)
    assert status.intact
    assert status.computed_sha256 == digest
    assert status.plan.controls == ("zero",)


def test_edited_plan_is_detected_and_makes_result_unreliable(tmp_path, noise_csv):
    path = tmp_path / "plan.json"
    save_plan(Plan(metric="hit", group_column="suite"), path)
    doc = json.loads(path.read_text())
    doc["alpha"] = 0.2  # loosened after seeing the data
    path.write_text(json.dumps(doc))
    status = load_plan(path)
    assert not status.intact
    ds = bn.load(noise_csv, metric="hit", group="suite")
    result = bn.check(ds, plan=status, n_resamples=2000)
    assert result.check("plan").status == "fail"
    assert result.verdict == "unreliable"


def test_plan_rejects_real_as_control():
    with pytest.raises(PlanError):
        Plan(metric="hit", real="real", controls=("real",))


def test_cli_check_writes_reports_and_returns_exit_code(tmp_path, noise_csv, capsys):
    html, js = tmp_path / "r.html", tmp_path / "r.json"
    code = main(["check", str(noise_csv), "--metric", "hit", "--group", "suite",
                 "--resamples", "2000", "--report", str(html), "--json", str(js)])
    assert code == 1
    out = capsys.readouterr().out
    assert "DOES NOT BEAT CONTROLS" in out
    page = html.read_text()
    assert page.startswith("<!doctype html>")
    assert 'src="http' not in page and 'href="http' not in page  # no remote assets
    assert json.loads(js.read_text())["verdict"] == "does-not-beat-controls"


def test_cli_plan_refuses_to_overwrite(tmp_path, capsys):
    out = tmp_path / "plan.json"
    assert main(["plan", "--metric", "hit", "-o", str(out)]) == 0
    assert main(["plan", "--metric", "hit", "-o", str(out)]) == 3
    assert "never overwritten" in capsys.readouterr().err


def test_cli_check_with_plan(tmp_path, noise_csv):
    plan = tmp_path / "plan.json"
    main(["plan", "--metric", "hit", "--group", "suite", "-o", str(plan)])
    assert main(["check", str(noise_csv), "--plan", str(plan), "--resamples", "2000"]) == 1


def test_cli_bad_column_is_exit_3(noise_csv, capsys):
    assert main(["check", str(noise_csv), "--metric", "nope"]) == 3
    assert "available columns" in capsys.readouterr().err


def test_cli_demo(tmp_path, capsys):
    out = tmp_path / "signal.csv"
    assert main(["demo", "signal", "-o", str(out), "--report", str(tmp_path / "s.html")]) == 0
    assert "BEATS CONTROLS" in capsys.readouterr().out


def test_cli_looks_table(capsys):
    assert main(["looks", "--max-looks", "3"]) == 0
    assert "false positive rate" in capsys.readouterr().out
