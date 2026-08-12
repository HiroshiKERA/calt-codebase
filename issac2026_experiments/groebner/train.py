"""Train a model for Groebner basis task: F=[f1,f2] -> G (Groebner basis).

将来的に Transformer-GB の設定に合わせて monomial order 変換や
特殊なエンコーディングを追加するが、ここでは最小の雛形だけ実装する。
"""

import os
from typing import Any

import click
from omegaconf import OmegaConf
from sage.all import GF, QQ, RR, ZZ, PolynomialRing  # type: ignore

from calt.io import (
    ChainLoadPreprocessor,
    ExpandedFormLoadPreprocessor,
    IOPipeline,
    TextToSageLoadPreprocessor,
)
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings


class _GroebnerLexOrderPreprocessor:
    """F,G (Sage 多項式リスト) を lex 順序に変換し、lex Groebner 基底を再計算する load preprocessor。

    - 入力: dict {"problem": [f1,f2,...], "solution": [...]}
    - 出力: (input_text, target_text) 文字列
        input_text: F_lex を ' | ' で連結した文字列
        target_text: G_lex (lex Groebner basis) を ' | ' で連結した文字列

    emit_objects=True のときは文字列ではなく dict {"problem", "answer"} を返す。
    ExpandedFormLoadPreprocessor をこの後ろに繋ぐ場合に使う（Section 5.5 の
    monomial embedding は C/E expanded form を要求するため）。
    """

    def __init__(self, ring_src, delimiter: str = "|", emit_objects: bool = False):
        self.R_src = ring_src
        base = ring_src.base_ring()
        names = ring_src.variable_names()
        self.R_lex = PolynomialRing(base, names, order="lex")
        self.delimiter = delimiter
        self.emit_objects = emit_objects

    def process_sample(self, source: dict[str, Any]) -> tuple[str, str] | dict[str, Any]:
        if not isinstance(source, dict):
            raise TypeError(
                f"_GroebnerLexOrderPreprocessor expects dict source, got {type(source).__name__}"
            )
        F_src = source.get("problem") or []
        # solution 側は無視し、lex で Groebner 基底を取り直す
        F_lex = [self.R_lex(f) for f in F_src]
        I_lex = self.R_lex.ideal(F_lex)
        G_lex = list(I_lex.groebner_basis())

        if self.emit_objects:
            return {"problem": F_lex, "answer": G_lex}

        sep = f" {self.delimiter} "
        input_text = sep.join(str(f) for f in F_lex)
        target_text = sep.join(str(g) for g in G_lex)
        return input_text, target_text


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/train.yaml",
    help="Path to train config YAML (model, train, data).",
)
@click.option(
    "--dryrun", is_flag=True, help="Run in dryrun mode with reduced settings."
)
@click.option(
    "--data_config_path",
    type=click.Path(exists=True),
    default="configs/data.yaml",
    help="Path to data config YAML (sampler, dataset).",
)
@click.option(
    "--training_order",
    type=click.Choice(["degrevlex", "lex"]),
    default="degrevlex",
    help="Which monomial order to train on. "
    "'degrevlex' uses dataset as-is, 'lex' converts F,G to lex order and recomputes GB.",
)
@click.option(
    "--expanded_form",
    is_flag=True,
    help="Feed C/E expanded form ('C<c> E<e1> E<e2>' terms joined by '+', "
    "polynomials joined by '|') instead of raw polynomial strings. Required by "
    "the monomial token embedding of Section 5.5; use configs/lexer_ce.yaml.",
)
def main(
    config_path: str,
    dryrun: bool,
    data_config_path: str,
    training_order: str,
    expanded_form: bool,
) -> None:
    """Train a model for Groebner basis task."""
    cfg = OmegaConf.load(config_path)

    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    # training_order ごとに保存先を分ける
    suffix = training_order
    save_dir = save_dir.rstrip("/") + f"_{suffix}"
    cfg.train.save_dir = save_dir

    # wandb の run name も training_order で区別する
    if (
        hasattr(cfg.train, "wandb")
        and hasattr(cfg.train.wandb, "name")
        and cfg.train.wandb.name
    ):
        base_name = cfg.train.wandb.name
        cfg.train.wandb.name = f"{base_name}_{suffix}"

    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)

    # データロード時の前処理は二つの理由で必要になる:
    #   training_order='lex' -> degrevlex -> lex の ring 変換 + GB 再計算
    #   expanded_form        -> 生の多項式文字列 -> C/E expanded form
    # どちらの場合もまず テキスト -> Sage 多項式 (TextToSageLoadPreprocessor)。
    if training_order == "lex" or expanded_form:
        data_cfg = OmegaConf.load(data_config_path)
        sampler_cfg = dict(OmegaConf.to_container(data_cfg.sampler, resolve=True))
        symbols = sampler_cfg.get("symbols", "x,y")
        field_str = sampler_cfg.get("field_str", "QQ")
        order = sampler_cfg.get("order", "degrevlex")

        # sampler で使っている ring（degrevlex）を再構築
        if field_str == "QQ":
            field = QQ
        elif field_str == "RR":
            field = RR
        elif field_str == "ZZ":
            field = ZZ
        elif field_str.startswith("GF"):
            p = int(field_str[2:]) if field_str[2:].isdigit() else None
            if not p:
                raise ValueError(f"Unsupported field_str for GF: {field_str!r}")
            field = GF(p)
        else:
            raise ValueError(f"Unsupported field_str: {field_str!r}")
        names = [s.strip() for s in symbols.split(",")]
        R_src = PolynomialRing(field, names, order=order)

        text_to_sage = TextToSageLoadPreprocessor(delimiter="|", ring=R_src)
        steps = [text_to_sage]
        if training_order == "lex":
            # expanded form を後段に繋ぐ場合は文字列ではなく Sage オブジェクトを渡す
            steps.append(
                _GroebnerLexOrderPreprocessor(
                    R_src, delimiter="|", emit_objects=expanded_form
                )
            )
        if expanded_form:
            # polynomial_multiplication / polynomial_reduction と同じ ' | ' 区切り
            steps.append(ExpandedFormLoadPreprocessor(delimiter=" | "))
        io_pipeline.dataset_load_preprocessor = ChainLoadPreprocessor(*steps)

    io_dict = io_pipeline.build()

    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    success_rate = trainer_pipeline.evaluate_and_save_generation()
    print(f"Success rate: {100 * success_rate:.1f}%")


if __name__ == "__main__":
    main()
