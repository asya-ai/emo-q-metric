import csv
import os
import random
import time

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from google import genai
from loguru import logger
from tqdm import tqdm
from pydantic import BaseModel, Field


class GenMOSMetric(BaseModel):
    speech_content_clarity_value: float = Field(
        ge=1.0,
        le=5.0,
        description="Speech content clarity score (How interpretable is the spoken content)."
    )
    audio_quality_value: float = Field(
        ge=1.0,
        le=5.0,
        description="Audio quality score (How good is the general quality of the audio)."
    )


if __name__ == "__main__":
    api_key = ''
    model_name = "gemini-3-pro-preview"
    client = genai.Client(api_key=api_key)

    root_path = "/media/storage_2/data/raw/tts_validation/en/styletts2_tts_valid_output"
    df_path = f"{root_path}/metadata.csv"
    df = pd.read_csv(df_path, sep=",")
    df = df.reindex(columns=df.columns.tolist() + ['gmos_score_speech_clarity'])
    df = df.reindex(columns=df.columns.tolist() + ['gmos_score_audio_quality'])
    df['gmos_score_speech_clarity'] = 1.0
    df['gmos_score_audio_quality'] = 1.0

    total_hours = 0.0
    last_processed_idx = 0
    for idx, row in tqdm(df.iterrows()):
        transcription = row[1].strip()
        file_name_original = row[0]

        if idx <= last_processed_idx:
            logger.info(f"Skipping index {idx} as it has already been processed")
            continue

        file_source_path = f"{root_path}/{file_name_original}"
        my_wav_file = client.files.upload(file=file_source_path)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=
                [
                    """
                    You will receive an audio speech sample.
                    Your task is to rate the spoken content clarity and overall audio quality.
                    """,
                    my_wav_file
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": GenMOSMetric.model_json_schema(),
                }
            )
            mos_metrics = GenMOSMetric.model_validate_json(response.text)
            speech_content_clarity_value = mos_metrics.speech_content_clarity_value
            audio_quality_value = mos_metrics.audio_quality_value
        except Exception as e:
            logger.error(e)
            speech_content_clarity_value = 1.0
            audio_quality_value = 1.0

        print(speech_content_clarity_value)
        print(audio_quality_value)
        df.at[idx, 'gmos_score_speech_clarity'] = float(speech_content_clarity_value)
        df.at[idx, 'gmos_score_audio_quality'] = float(audio_quality_value)

    df.to_csv(f"{root_path}/metadata_gmos.csv", index=False)

    """
    Final score:
    gmos = 0.5 * (speech_content_clarity_value + audio_quality_value)
    """
