import argparse
import os

import pandas as pd
from loguru import logger
from tqdm import tqdm
from typing import List, Tuple
import librosa

import torch
import torch.utils.data
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline


class EmoProcess:
    def __init__(
            self,
            args,
            datasource_samplerate=16000,
            step_sample_length=1 * 16000,
            window_sample_length=4 * 16000,
            model_path="./ckpt/checkpoint-3684",
            is_debug=False
    ):
        self.args = args
        self.is_debug = is_debug
        self.datasource_samplerate = datasource_samplerate
        self.step_sample_length = step_sample_length
        self.window_sample_length = window_sample_length
        self.batch_size = 4
        self.emotion_classes = ["happiness", "anger", "sadness", "other"]

        self.emotions_input_size = 64000
        self.emotion_presence_threshold = 0.1  # Threshold for considering emotion presence in given speech segment

        emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
        emotion_model = AutoModelForAudioClassification.from_pretrained(model_path, local_files_only=True).to("cuda")
        self.emotion_detection_pipeline = pipeline(
            "audio-classification",
            model=emotion_model,
            feature_extractor=emotion_feature_extractor,
            batch_size=self.batch_size
        )

    def normalize_audio(self, raw_x):
        raw_x = np.array(raw_x)
        x_min = np.amin(raw_x)
        x_max = np.amax(raw_x)
        eps = 1e-8
        raw_x = ((raw_x - x_min + eps) / (x_max - x_min + eps) - 0.5) * 2.0
        return raw_x

    def extract_start_end(self, y_segment, sr=16000) -> Tuple[List[float], List[float]]:
        list_time_start_sec = []
        list_time_end_sec = []
        try:
            y_parts = []
            y_segment = self.normalize_audio(y_segment)

            # Only 4 sec for new model
            if len(y_segment) > self.window_sample_length:
                for i in range(0, len(y_segment), self.step_sample_length):
                    y_chunk = y_segment[i:i + self.window_sample_length]
                    if len(y_chunk) >= self.step_sample_length:
                        if len(y_chunk) < self.window_sample_length:
                            y_chunk_padded = np.pad(y_chunk, (0, self.window_sample_length - len(y_chunk)), 'constant')
                            y_parts.append(y_chunk_padded)
                        else:
                            y_parts.append(y_chunk)
                    else:
                        break
            else:
                # Pad to 4sec at 16kHz
                y_segment_padded = np.pad(y_segment, (0, self.window_sample_length - len(y_segment)), 'constant')
                y_parts = [y_segment_padded]

            time_sec = 0
            for y in tqdm(y_parts, desc="extracting time"):
                y_min = np.min(y)
                y_max = np.max(y)
                y_len_sec = min(self.step_sample_length / sr, len(y) / sr)

                if y_max > y_min:
                    list_time_start_sec.append(time_sec)
                    list_time_end_sec.append(time_sec + y_len_sec)
                time_sec += y_len_sec

        except Exception as exc:
            logger.exception(exc)
        return list_time_start_sec, list_time_end_sec

    def process_y(
            self,
            y_segment,
            sr=16000
    ) -> str:
        out_emo_label = 'no_emotion'
        try:
            list_time_start_sec, list_time_end_sec = self.extract_start_end(y_segment)

            y_emo_cce_parts = None
            y_parts_all = []
            for start_time_sec, end_time_sec in zip(list_time_start_sec, list_time_end_sec):
                y_part_pad_emo = np.zeros((self.emotions_input_size,), dtype=np.float32)
                y_part_emo = y_segment[int(sr * start_time_sec):int(sr * start_time_sec) + self.emotions_input_size]
                y_part_pad_emo[:len(y_part_emo)] = y_part_emo
                y_parts_all.append(torch.FloatTensor(y_part_pad_emo))

            t_parts_all = torch.stack(y_parts_all)
            if len(t_parts_all) > 1:
                batcher_emo = torch.utils.data.DataLoader(t_parts_all, batch_size=1, shuffle=False)
                y_emo_cce_parts = []
                for batch_emotions in tqdm(batcher_emo, 'emo_cce'):
                    in_batch = [b.numpy() for b in batch_emotions]
                    results = self.emotion_detection_pipeline(in_batch)
                    emo_label = results[0][0]['label']

                    if results[0][1]['score'] > self.emotion_presence_threshold and results[0][1]['label'] != 'no_emotion':
                        emo_label = results[0][1]['label']

                    y_emo_cce_parts.append(emo_label)

            if y_emo_cce_parts is not None:
                for y_emo_cce_class in y_emo_cce_parts:
                    if y_emo_cce_class != 'no_emotion':
                        out_emo_label = y_emo_cce_class
                        break

            with torch.no_grad():
                torch.cuda.empty_cache()

        except Exception as exc:
            logger.exception(exc)

        return out_emo_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-model_path', default='./ckpt/checkpoint-3684', type=str)
    parser.add_argument('-datasource_samplerate', default=16000, type=int)
    parser.add_argument('-step_sample_length', default=1 * 16000, type=int)
    parser.add_argument('-window_sample_length', default=4 * 16000, type=int)  # 4sec as used in model training

    args = parser.parse_args()

    emotion_processor = EmoProcess(args, is_debug=False, model_path=args.model_path)


    ### EXAMPLE inference from data paths that contain audio .wav files and metadata.csv in huggingface audiofolder data format

    data_paths = [
        './',
    ]

    total_process_count = 20000
    cur_progress = 0

    all_dataset_stats = {}
    processed_rows = []
    for data_path in data_paths:
        df = pd.read_csv(os.path.join(data_path, 'metadata.csv'))

        emo_data_labels = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            if cur_progress >= total_process_count:
                break

            try:
                file_path = os.path.join(data_path, row['file_name'])
                y, sr = librosa.load(file_path, sr=16000)
                emo_label = emotion_processor.process_y(y)
                emo_data_labels.append(emo_label)

                cur_progress += 1
                processed_rows.append(row)
            except Exception as e:
                logger.exception(e)
                emo_data_labels.append('no_emotion')


        df_out = pd.DataFrame(processed_rows)
        df_out['emo_label'] = emo_data_labels
        df_out.to_csv(f'{data_path}/metadata_emo.csv', index=False)

        print(f'Statistics for {data_path}:')
        print(df_out['emo_label'].value_counts())
        all_dataset_stats[data_path] = df_out['emo_label'].value_counts().to_dict()
