import pytest
pd = pytest.importorskip("pandas")

from entropy_news.main import main as train_main
from entropy_news.main_forecast import main as forecast_main
from entropy_news.main_evaluate import main as eval_main


def _write_dummy_glove(path):
    path.write_text("""hello 0.1 0.2
world 0.3 0.4
again 0.5 0.6
<unk> 0.0 0.0
""")


def test_training_and_forecasting(tmp_path):
    torch = pytest.importorskip("torch")

    train_file = tmp_path / "train.txt"
    train_file.write_text("hello world\nhello again\n")
    new_file = tmp_path / "new.txt"
    new_file.write_text("hello world\n")
    glove = tmp_path / "glove.txt"
    _write_dummy_glove(glove)

    model_out = tmp_path / "model.pth"
    vocab_out = tmp_path / "vocab.json"

    train_main([
        "--train-data",
        str(train_file),
        "--glove-path",
        str(glove),
        "--embed-dim",
        "2",
        "--hidden-dim",
        "2",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--model-out",
        str(model_out),
        "--vocab-out",
        str(vocab_out),
    ])

    assert model_out.exists()
    assert vocab_out.exists()

    out_csv = tmp_path / "forecast.csv"
    forecast_main([
        "--vocab-path",
        str(vocab_out),
        "--model-path",
        str(model_out),
        "--new-data",
        str(new_file),
        "--output-csv",
        str(out_csv),
        "--embed-dim",
        "2",
        "--hidden-dim",
        "2",
        "--fine-tune-epochs",
        "1",
    ])

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert {"ENT", "ENT_news", "ENT_model"}.issubset(df.columns)


def test_full_evaluation_flow(tmp_path) -> None:
    """Run training then evaluation CLI end-to-end."""

    torch = pytest.importorskip("torch")

    train_file = tmp_path / "train.txt"
    train_file.write_text("hello world\nhello again\n")
    eval_file = tmp_path / "eval.txt"
    eval_file.write_text("hello again\n")
    glove = tmp_path / "glove.txt"
    _write_dummy_glove(glove)

    model_out = tmp_path / "model.pth"
    vocab_out = tmp_path / "vocab.json"

    train_main([
        "--train-data",
        str(train_file),
        "--glove-path",
        str(glove),
        "--embed-dim",
        "2",
        "--hidden-dim",
        "2",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--model-out",
        str(model_out),
        "--vocab-out",
        str(vocab_out),
    ])

    out_csv = tmp_path / "eval.csv"
    eval_main([
        "--vocab-path",
        str(vocab_out),
        "--model-path",
        str(model_out),
        "--data",
        str(eval_file),
        "--output-csv",
        str(out_csv),
        "--embed-dim",
        "2",
        "--hidden-dim",
        "2",
    ])

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert {"entropy", "perplexity"}.issubset(df.columns)


def test_resume_training(tmp_path) -> None:
    """Train for one epoch, resume for a second, and update the checkpoint."""

    torch = pytest.importorskip("torch")

    train_file = tmp_path / "train.txt"
    train_file.write_text("hello world\nhello again\n")
    glove = tmp_path / "glove.txt"
    _write_dummy_glove(glove)

    checkpoint = tmp_path / "ckpt.pth"
    model_out = tmp_path / "model.pth"
    vocab_out = tmp_path / "vocab.json"

    # First run creates the checkpoint at epoch 1
    train_main([
        "--train-data",
        str(train_file),
        "--glove-path",
        str(glove),
        "--embed-dim",
        "2",
        "--hidden-dim",
        "2",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--model-out",
        str(model_out),
        "--vocab-out",
        str(vocab_out),
        "--checkpoint",
        str(checkpoint),
    ])

    assert checkpoint.exists()
    ckpt = torch.load(checkpoint)
    assert ckpt["epoch"] == 1

    # Resume training to complete second epoch
    train_main([
        "--train-data",
        str(train_file),
        "--glove-path",
        str(glove),
        "--embed-dim",
        "2",
        "--hidden-dim",
        "2",
        "--epochs",
        "2",
        "--batch-size",
        "1",
        "--model-out",
        str(model_out),
        "--vocab-out",
        str(vocab_out),
        "--checkpoint",
        str(checkpoint),
        "--resume-from",
        str(checkpoint),
    ])

    ckpt = torch.load(checkpoint)
    assert ckpt["epoch"] == 2
