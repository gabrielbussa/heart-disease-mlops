import pytest
from heart_mlops.data import validate_dataset, FEATURE_COLUMNS, TARGET_COLUMN

def test_validate_accepts_binary_dataset(synthetic_frame):
    validate_dataset(synthetic_frame)

def test_validate_rejects_missing_column(synthetic_frame):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(synthetic_frame.drop(columns=[FEATURE_COLUMNS[0]]))

def test_validate_rejects_non_binary_target(synthetic_frame):
    frame = synthetic_frame.copy()
    frame.loc[frame.index[0], TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="only 0/1"):
        validate_dataset(frame)

def test_download_uses_fallback_when_primary_fails(monkeypatch, tmp_path):
    from heart_mlops import data as data_module
    rows = []
    for i in range(260):
        rows.append("63,1,1,145,233,1,2,150,0,2.3,3,0,6,0")
    csv_text = "\n".join(rows)
    calls = []

    class FakeResponse:
        def __init__(self, text):
            self.text = text
            self.content = text.encode()
        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("primary host unavailable")
        return FakeResponse(csv_text)

    monkeypatch.setattr(data_module.requests, "get", fake_get)
    destination = tmp_path / "raw.data"
    result = data_module.download_dataset(destination)
    assert result.exists()
    assert len(calls) == 2
    assert len(result.read_text().splitlines()) == 260
