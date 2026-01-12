# Emo-Q-metric
Repository for emotion quality (EmoQ) detection model used in TTS generated speech assessment.

### Emotion Training dataset stats
| Emotion    | Count |
|:-----------|------:|
| other      | 6156  |
| anger      | 6094  |
| happiness  | 6026  |
| sadness    | 5960  |

![Emotion Label Distribution](assets/emotion_label_distribution.png)

Model was trained in binary emotion classifier format to detect with confidence value if the speech sample contains emotional content or not.
Data was structured in 2 classes [happiness, anger, sadness] vs [other] -> **emotion** | **no_emotion** classes


# Model inference

- Weights are automatically downloaded from Huggingface: https://huggingface.co/asya-ai/Emo-Q-Wav2vec2
