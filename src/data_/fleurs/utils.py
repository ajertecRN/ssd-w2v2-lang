import os
from pathlib import Path
import shutil
from typing import Tuple

from datasets import load_dataset
import pandas as pd

# According to https://huggingface.co/datasets/google/xtreme_s/commit/
# 67c9d395b03d68600af87ff30601f445fcb6db35
FLEURS_LANGUAGE_CODES = {
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
}


def sample_to_row(sample) -> Tuple:
    return (
        int(sample["id"]),
        str(Path(sample["path"]).resolve()),
        str(Path(sample["path"]).resolve().name),
        str(sample["raw_transcription"]),
        str(sample["transcription"]),
        "male"
        if sample["gender"] == 0
        else "female"
        if sample["gender"] == 1
        else None,
    )


def aggregate_samples(samples) -> pd.DataFrame:
    rows = tuple(sample_to_row(sample) for sample in samples)
    df = (
        pd.DataFrame(
            rows,
            columns=["id", "path", "name", "transcript_raw", "transcript", "gender"],
        )
        .dropna(axis=0, how="any", subset=["id", "name", "gender"])
        .astype(
            {
                "id": int,
                "path": str,
                "name": str,
                "transcript_raw": str,
                "transcript": str,
                "gender": str,
            }
        )
        .reset_index(drop=True)
    )

    return df


def fleurs_hf_to_df(fleurs_hf):
    train = aggregate_samples(samples=fleurs_hf["train"])
    dev = aggregate_samples(samples=fleurs_hf["validation"])
    test = aggregate_samples(samples=fleurs_hf["test"])

    train["split"] = "train"
    dev["split"] = "dev"
    test["split"] = "test"

    df = (
        pd.concat([train, dev, test], axis=0)[
            ["id", "path", "name", "split", "transcript_raw", "transcript", "gender"]
        ]
        .dropna()
        .drop_duplicates()
        .sort_values(
            ["id", "path", "name", "split", "transcript_raw", "transcript", "gender"]
        )
        .reset_index(drop=True)
    )

    return df


def normalize_fleurs_df(fleurs_df: pd.DataFrame):
    transcripts_df = (
        fleurs_df[["id", "transcript_raw", "transcript"]]
        .dropna()
        .drop_duplicates("id")
        .sort_values("id")
        .reset_index(drop=True)
    )
    recordings_df = (
        fleurs_df[["id", "name", "split", "gender"]]
        .dropna()
        .drop_duplicates(["id", "name"])
        .sort_values(["id", "name", "split", "gender"])
        .reset_index(drop=True)
    )

    return transcripts_df, recordings_df


def assemble_dataset(
    dataset_dir: Path, language_code: str = "en_us", clean_up: bool = False
):
    language_code = str(language_code).strip().lower()
    dataset_dir = Path(dataset_dir).resolve()

    cache_dir = dataset_dir / "temp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fleurs_hf = load_dataset(
        path="google/xtreme_s",
        name=f"fleurs.{language_code}",
        cache_dir=f"{cache_dir}",
    )
    fleurs_df = fleurs_hf_to_df(fleurs_hf=fleurs_hf)

    # Copy over audio data
    audio_dir = dataset_dir / "audio" / "original"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for path in fleurs_df["path"].dropna().unique():
        path = Path(path).resolve()
        if path.exists():
            shutil.copyfile(str(path), str(audio_dir / path.name))

    # Copy over data
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    transcripts_df, recordings_df = normalize_fleurs_df(fleurs_df=fleurs_df)
    transcripts_df.to_csv(data_dir / "transcripts.csv", index=False)
    recordings_df.to_csv(data_dir / "recordings.csv", index=False)

    # Delete cache folder
    if clean_up:
        shutil.rmtree(str(cache_dir))
