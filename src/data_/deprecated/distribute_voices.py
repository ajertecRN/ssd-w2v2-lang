import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_VENDOR_TO_PROPORTION = {"amazon": 0.2, "azure": 0.4, "google": 0.4}
VITS_VENDOR_TO_PROPORTION = {
    "coqui": 1.0,
}


def get_vendor_probabilities(df, vendor_to_proportion, splits):
    splits = tuple(set(x for x in splits if x))

    identifier = ["tts_vendor", "tts_engine"]
    identifier = [x for x in identifier if x in df.columns]
    assert len(identifier) > 0, "No identifier columns found in DataFrame"

    df_for_splits = df[df["tts_split"].isin(splits)].drop_duplicates()
    remaining_vendors = df_for_splits["tts_vendor"].unique()
    vendor_to_proportion = {
        v: p for v, p in vendor_to_proportion.items() if v in remaining_vendors
    }
    vendor_to_proportion = {
        v: p / sum(vendor_to_proportion.values())
        for v, p in vendor_to_proportion.items()
    }

    vendor_counts = df_for_splits["tts_vendor"].value_counts()
    vendor_probability = tuple(
        (vendor, proportion / vendor_counts[vendor])
        for vendor, proportion in vendor_to_proportion.items()
        if vendor in vendor_counts
    )

    return vendor_probability


def vendor_probabilities_to_df(vendor_probabilities, split=None):
    df = pd.DataFrame(vendor_probabilities, columns=["tts_vendor", "probability"])
    df = df.astype({"tts_vendor": str, "probability": float})

    if split:
        df["tts_split"] = str(split)

    return df


def get_default_vendor_probabilities_df(df, vendor_to_proportion):
    train_dev_vendor_probabilities = get_vendor_probabilities(
        df=df,
        vendor_to_proportion=vendor_to_proportion,
        splits=["train_dev"],
    )
    test_vendor_probabilities = get_vendor_probabilities(
        df=df,
        vendor_to_proportion=vendor_to_proportion,
        splits=["test"],
    )

    vendor_probabilities_df = pd.concat(
        [
            vendor_probabilities_to_df(
                vendor_probabilities=train_dev_vendor_probabilities, split="train_dev"
            ),
            vendor_probabilities_to_df(
                vendor_probabilities=test_vendor_probabilities, split="test"
            ),
        ],
        axis=0,
    )

    return vendor_probabilities_df


def generate_speaker_distribution(
    dataset_df,
    speakers_df,
    vendor_probabilities_df,
):
    speakers_df = pd.merge(
        left=speakers_df,
        right=vendor_probabilities_df,
        how="left",
        left_on=("tts_vendor", "tts_split"),
        right_on=("tts_vendor", "tts_split"),
    )

    chunks = list()
    id_column = "id"
    fleurs_ids = dataset_df[id_column].unique().tolist()
    for fleurs_id in tqdm(
        fleurs_ids,
        desc="Distributing speakers",
        total=len(fleurs_ids),
        file=sys.stdout,
        ncols=80,
        unit="S",
        unit_scale=True,
    ):
        sample = pd.DataFrame(dataset_df[dataset_df[id_column] == fleurs_id])
        male_sample = sample[sample["gender"] == "male"].reset_index()
        female_sample = sample[sample["gender"] == "female"].reset_index()

        sample_split = sample["split"].values.tolist()[0]
        speaker_split = (
            "train_dev" if str(sample_split).lower() in ("train", "dev") else "test"
        )
        relevant_speakers = speakers_df[speakers_df["tts_split"] == speaker_split]

        male_selection = list()
        female_selection = list()
        if relevant_speakers.shape[0] > 0:
            vendor_groups = relevant_speakers[
                ["tts_vendor", "tts_group", "probability"]
            ].drop_duplicates()
            probabilities = vendor_groups["probability"]
            vendor_groups = vendor_groups.drop("probability", axis=1)
            relevant_speakers = relevant_speakers.drop("probability", axis=1)

            vendor_groups_draft = vendor_groups.sample(
                n=vendor_groups.shape[0], replace=False, weights=probabilities
            )

            chosen_speakers = list()
            for vendor, group in vendor_groups_draft.itertuples(index=False):
                relevant_group = relevant_speakers[
                    (relevant_speakers["tts_vendor"] == vendor)
                    & (relevant_speakers["tts_group"] == group)
                ]
                chosen_speakers.append(relevant_group.sample(1))
            chosen_speakers = pd.concat(chosen_speakers, axis=0)

            male_draft = chosen_speakers[
                chosen_speakers["tts_gender"] == "male"
            ].values.tolist()
            female_draft = chosen_speakers[
                chosen_speakers["tts_gender"] == "female"
            ].values.tolist()

            male_selection.extend(male_draft[: male_sample.shape[0]])
            female_selection.extend(female_draft[: female_sample.shape[0]])

            n_male_missing = male_sample.shape[0] - len(male_selection)
            male_selection.extend(
                female_draft[
                    len(female_selection) : len(female_selection) + n_male_missing
                ]
            )

            n_female_missing = female_sample.shape[0] - len(female_selection)
            female_selection.extend(
                male_draft[len(male_selection) : len(male_selection) + n_female_missing]
            )

        male_selection_df = pd.DataFrame(
            male_selection,
            columns=[
                "tts_vendor",
                "tts_name",
                "tts_engine",
                "tts_split",
                "tts_gender",
                "tts_group",
            ],
        )
        female_selection_df = pd.DataFrame(
            female_selection,
            columns=[
                "tts_vendor",
                "tts_name",
                "tts_engine",
                "tts_split",
                "tts_gender",
                "tts_group",
            ],
        )

        male_rows = pd.concat([male_sample, male_selection_df], axis=1)
        female_rows = pd.concat([female_sample, female_selection_df], axis=1)

        chunks.append(pd.concat([male_rows, female_rows], axis=0))

    rows = pd.concat(chunks, axis=0)[
        ["id", "name", "tts_vendor", "tts_name", "tts_engine", "gender", "tts_gender"]
    ]
    rows = rows.sort_values(list(rows.columns))

    return rows


def analyze_speaker_distribution(dataset_df, speaker_distribution):
    id_and_split = dataset_df[["id", "split"]].drop_duplicates()
    speaker_distribution = pd.merge(
        speaker_distribution, id_and_split, how="left", left_on="id", right_on="id"
    )

    for split in sorted(speaker_distribution["split"].unique()):
        print(f"{split}:")
        current_subset = speaker_distribution[
            speaker_distribution["split"] == split
        ].drop(["id", "name"], axis=1)

        vendor_value_counts = current_subset["tts_vendor"].value_counts(normalize=True)
        male_subset = current_subset[current_subset["tts_gender"] == "male"]
        female_subset = current_subset[current_subset["tts_gender"] == "female"]

        male_subset = male_subset[["tts_name", "tts_engine"]]
        female_subset = female_subset[["tts_name", "tts_engine"]]

        print(vendor_value_counts.to_string())
        print()


def distribute_voices(
    recordings_path,
    voices_path,
    vendor_to_proportion,
    result_path,
    drop_gender_mismatch=False,
    drop_unassigned=False,
):
    dataset_df = pd.read_csv(
        recordings_path,
        usecols=["id", "name", "split", "gender"],
    )

    speakers_df = pd.read_csv(
        voices_path,
        usecols=[
            "tts_vendor",
            "tts_name",
            "tts_engine",
            "tts_split",
            "tts_gender",
            "tts_group",
        ],
    )
    speakers_df = speakers_df[speakers_df["tts_vendor"].isin(vendor_to_proportion)]

    speakers_df = speakers_df.sort_values(list(speakers_df.columns))

    vendor_probabilities_df = get_default_vendor_probabilities_df(
        df=speakers_df, vendor_to_proportion=vendor_to_proportion
    )

    speaker_distribution = generate_speaker_distribution(
        dataset_df=dataset_df,
        speakers_df=speakers_df,
        vendor_probabilities_df=vendor_probabilities_df,
    ).reset_index(drop=True)
    print(f"Distributed TTS speakers; {speaker_distribution.shape[0]} in total")

    if not drop_unassigned:
        tts_columns = sorted(
            set(speaker_distribution.columns) - set(("id", "name", "split", "gender"))
        )
        speaker_distribution = speaker_distribution.dropna(
            axis=0,
            how="all",
            subset=tts_columns,
        )

        print(f"Dropped null voices; {speaker_distribution.shape[0]} remaining")
    if drop_gender_mismatch:
        gender_mismatch = (
            speaker_distribution["gender"] != speaker_distribution["tts_gender"]
        )
        gender_mismatch_index = speaker_distribution[gender_mismatch].index
        speaker_distribution = speaker_distribution.drop(gender_mismatch_index)

        print(f"Dropped gender mismatches; {speaker_distribution.shape[0]} remaining")

    speaker_distribution = (
        speaker_distribution.fillna(
            ["name", "tts_vendor", "tts_name", "tts_engine", "gender", "tts_gender"]
        )
        .astype(
            {
                "id": int,
                "name": str,
                "tts_vendor": str,
                "tts_name": str,
                "tts_engine": str,
                "gender": str,
                "tts_gender": str,
            }
        )
        .sort_values(list(speaker_distribution.columns))
        .reset_index(drop=True)
    )

    print()
    analyze_speaker_distribution(
        dataset_df=dataset_df, speaker_distribution=speaker_distribution
    )

    speaker_distribution.to_csv(result_path, index=False)


def get_parser():
    parser = argparse.ArgumentParser()

    inputs = parser.add_argument_group("Inputs")
    outputs = parser.add_argument_group("Outputs")

    inputs.add_argument(
        "--recordings_path",
        type=str,
        required=True,
        metavar="FILEPATH",
        help="Path to the recordings CSV file",
    )
    inputs.add_argument(
        "--voices_path",
        type=str,
        required=True,
        metavar="FILEPATH",
        help="Path to the voices CSV file",
    )
    inputs.add_argument(
        "--mode",
        type=str,
        choices=["base", "vits"],
        default="base",
        help="Mode for vendor distributions. Default: base",
    )

    outputs.add_argument(
        "--result_path",
        type=str,
        required=True,
        metavar="FILEPATH",
        help="Path to the resulting CSV file",
    )
    outputs.add_argument(
        "--drop_gender_mismatch",
        action="store_true",
        help=("If set, will drop samples assigned voices with a non-matching gender"),
    )
    outputs.add_argument(
        "--drop_unassigned",
        action="store_true",
        help="If set, will drop samples which were not assigned a voice",
    )
    outputs.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="UINT32",
        help="The seed for RNG operations. Default: 0",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    pd.core.common.random_state(args.seed)
    np.random.seed(args.seed)

    distribute_voices(
        recordings_path=Path(args.recordings_path).resolve(),
        voices_path=Path(args.voices_path).resolve(),
        vendor_to_proportion=BASE_VENDOR_TO_PROPORTION
        if args.mode == "base"
        else VITS_VENDOR_TO_PROPORTION,
        result_path=args.result_path,
        drop_gender_mismatch=args.drop_gender_mismatch,
        drop_unassigned=args.drop_unassigned,
    )


if __name__ == "__main__":
    main()
