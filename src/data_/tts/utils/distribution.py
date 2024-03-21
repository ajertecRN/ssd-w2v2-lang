import re
import sys
from typing import Dict, Iterable, Optional

import pandas as pd
from tqdm import tqdm

VENDOR_TO_PROBABILITY_SEPARATOR_REGEX = re.compile(":\s*")


def normalize_vendor_to_probability(
    vendor_to_probability: Dict[str, float]
) -> Dict[str, float]:
    vendor_to_probability = {v: p for v, p in vendor_to_probability.items() if p > 0}

    total_probability = sum(vendor_to_probability.values())
    vendor_to_probability = {
        v: p / total_probability for v, p in vendor_to_probability.items()
    }

    return vendor_to_probability


def get_weighted_voices(
    voices_df: pd.DataFrame, vendor_to_probability: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    vendors = sorted(voices_df["tts_vendor"].dropna().unique())

    if vendor_to_probability is None:
        vendor_to_probability = {vendor: 1.0 for vendor in vendors}
    vendor_to_probability = normalize_vendor_to_probability(
        vendor_to_probability=vendor_to_probability
    )

    missing_vendors = set(vendors) - set(vendor_to_probability.keys())
    voices_df = voices_df[~voices_df["tts_vendor"].isin(missing_vendors)]

    vendor_weights = pd.DataFrame(
        list(vendor_to_probability.items()),
        columns=["tts_vendor", "probability"],
    ).astype({"tts_vendor": str, "probability": float})
    voice_groups_per_vendor = (
        voices_df.drop_duplicates(["tts_vendor", "tts_group"])
        .groupby("tts_vendor")["tts_group"]
        .size()
        .reset_index(name="n_groups")
    )
    vendor_weights = vendor_weights.merge(
        voice_groups_per_vendor,
        how="left",
        left_on="tts_vendor",
        right_on="tts_vendor",
    )
    vendor_weights["probability"] = (
        vendor_weights["probability"] / vendor_weights["n_groups"]
    )
    vendor_weights = vendor_weights.drop("n_groups", axis=1)

    voices_df = voices_df.merge(
        vendor_weights, how="left", left_on="tts_vendor", right_on="tts_vendor"
    )
    voices_df = (
        voices_df.dropna(axis=0, subset="probability")
        .drop_duplicates()
        .sort_values(list(voices_df.columns))
        .reset_index(drop=True)
    )

    return voices_df


def distribute_voice_groups(
    genders_df: pd.DataFrame,
    weighted_voices_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    ids = sorted(genders_df["id"].unique())
    if verbose:
        ids = tqdm(
            ids,
            desc="Distributing voice groups",
            total=len(ids),
            file=sys.stdout,
            ncols=80,
            unit="sent",
            unit_scale=True,
        )

    samples_with_voice_groups = list()
    for i in ids:
        speakers = pd.DataFrame(genders_df[genders_df["id"] == i])
        female_speakers = speakers[speakers["gender"] == "female"].reset_index(
            drop=True
        )
        male_speakers = speakers[speakers["gender"] == "male"].reset_index(drop=True)

        voice_groups = weighted_voices_df.sample(
            n=weighted_voices_df.shape[0],
            replace=False,
            weights=weighted_voices_df["probability"],
        )
        female_voice_groups = voice_groups[voice_groups["tts_gender"] == "female"][
            ["tts_vendor", "tts_group"]
        ].values.tolist()
        male_voice_groups = voice_groups[voice_groups["tts_gender"] == "male"][
            ["tts_vendor", "tts_group"]
        ].values.tolist()

        # Phase 1: Trying to match genders
        selected_female = female_voice_groups[: female_speakers.shape[0]]
        selected_male = male_voice_groups[: male_speakers.shape[0]]

        # Phase 2: Filling rest of voices with opposite gender
        selected_female += male_voice_groups[
            len(selected_male) : female_speakers.shape[0] - len(selected_female)
        ]
        selected_male += female_voice_groups[
            len(selected_female) : male_speakers.shape[0] - len(selected_male)
        ]

        samples_with_voice_groups += [
            pd.concat(
                [
                    female_speakers,
                    pd.DataFrame(selected_female, columns=["tts_vendor", "tts_group"]),
                ],
                axis=1,
            ),
            pd.concat(
                [
                    male_speakers,
                    pd.DataFrame(selected_male, columns=["tts_vendor", "tts_group"]),
                ],
                axis=1,
            ),
        ]

    samples_with_voice_groups = pd.concat(samples_with_voice_groups, axis=0)[
        ["id", "name", "tts_vendor", "tts_group"]
    ]
    samples_with_voice_groups = (
        samples_with_voice_groups.drop_duplicates()
        .sort_values(list(samples_with_voice_groups.columns))
        .reset_index(drop=True)
    )

    return samples_with_voice_groups


def resolve_groups_to_voices(
    voices_df: pd.DataFrame,
    voice_groups: pd.DataFrame,
) -> pd.DataFrame:
    voice_groups = voice_groups.reset_index(drop=True).reset_index(names="i")

    voices = (
        voice_groups.merge(
            voices_df,
            how="left",
            left_on=["tts_vendor", "tts_group"],
            right_on=["tts_vendor", "tts_group"],
        )
        .groupby(["i", "tts_vendor", "tts_group"], group_keys=False)
        .apply(lambda x: x.sample(1))
        .drop(["i", "tts_group"], axis=1)
        .reset_index(drop=True)
    )

    return voices


def distribute_voices(
    recordings_df: pd.DataFrame,
    voices_df: pd.DataFrame,
    vendor_to_probability: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    recordings_df = (
        recordings_df[["id", "name", "split", "gender"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["id", "name", "split", "gender"])
        .reset_index(drop=True)
    )

    voices_df = (
        voices_df[
            [
                "tts_vendor",
                "tts_name",
                "tts_engine",
                "tts_split",
                "tts_gender",
                "tts_group",
            ]
        ]
        .dropna(axis=0, how="all", subset=["tts_vendor", "tts_name", "tts_engine"])
        .dropna(axis=0, how="any", subset=["tts_split", "tts_gender", "tts_group"])
        .drop_duplicates()
        .sort_values(
            [
                "tts_vendor",
                "tts_name",
                "tts_engine",
                "tts_split",
                "tts_gender",
                "tts_group",
            ]
        )
        .reset_index(drop=True)
    )

    distribution = list()
    for tts_split in voices_df["tts_split"].unique():
        splits = str(tts_split).split("_")
        splits = [x for x in splits if x]

        genders_df = (
            recordings_df[recordings_df["split"].isin(splits)][["id", "name", "gender"]]
            .drop_duplicates()
            .sort_values(["id", "name", "gender"])
            .reset_index(drop=True)
        )
        weighted_voices_df = (
            get_weighted_voices(
                voices_df=voices_df[voices_df["tts_split"] == tts_split],
                vendor_to_probability=vendor_to_probability,
            )[["tts_vendor", "tts_group", "tts_gender", "probability"]]
            .drop_duplicates(subset=["tts_vendor", "tts_group"])
            .sort_values(["tts_vendor", "tts_group", "tts_gender", "probability"])
            .reset_index(drop=True)
        )

        samples_with_voice_groups = distribute_voice_groups(
            genders_df=genders_df,
            weighted_voices_df=weighted_voices_df,
            verbose=verbose,
        )
        samples_with_voices = resolve_groups_to_voices(
            voices_df=voices_df, voice_groups=samples_with_voice_groups
        )

        distribution.append(samples_with_voices)

    distribution = pd.concat(distribution, axis=0).drop("tts_split", axis=1)
    distribution = (
        distribution.drop_duplicates()
        .sort_values(list(distribution.columns))
        .reset_index(drop=True)
    )

    return distribution


def parse_vendor_to_probability(
    vendor_to_probability: Optional[Iterable[str]],
) -> Optional[Dict[str, float]]:
    if vendor_to_probability is None:
        return None

    vendor_to_probability = [str(vtp).rstrip() for vtp in vendor_to_probability]
    vendor_to_probability = [
        VENDOR_TO_PROBABILITY_SEPARATOR_REGEX.split(vtp)
        for vtp in vendor_to_probability
    ]
    vendor_to_probability = [
        vtp for vtp in vendor_to_probability if (vtp and len(vtp) == 2)
    ]
    vendor_to_probability = {str(v): float(p) for v, p in vendor_to_probability}

    return vendor_to_probability
