"""AlphaDefect のモデルを自分のマシンで動かすためのローカル実行系。

    python -m local status            何が揃っていて何が足りないか
    python -m local setup             ゲーム資産・重み・エンジンを用意する
    python -m local play --character DEFECT --seed ABC123 --asc 10
    python -m local selfcheck         エンジンとゲーム資産の噛み合わせを確認する
    python -m local versions          対応バージョンとキャラの一覧
    python -m local api --character DEFECT --steps 5   共通推論 API の動作例

ゲーム資産（Slay the Spire 2 本体）と重みは同梱しない。どちらも自分のマシンの
キャッシュ（既定 ~/.local/share/sls2、SLS2_HOME で変更）に置く。
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
