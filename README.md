## Results & Observations

- The model was fine-tuned using DistilBERT (multilingual) for primary emotion classification.
- Due to heavy class imbalance in the dataset, overall accuracy remains moderate.
- Weighted F1-score performs better than macro F1-score, indicating stronger performance on frequent emotion classes.
- Visualizations such as t-SNE embeddings, confusion matrix, and attention heatmaps were used to analyze model behavior.

## Limitations

- Several emotion labels had very few samples and were removed during preprocessing.
- Subtle emotional expressions are difficult to distinguish without larger datasets.
- The model occasionally predicts dominant emotional categories for ambiguous inputs.

## Future Improvements

- Collect balanced emotion datasets
- Apply data augmentation for low-frequency emotions
- Use hierarchical emotion classification or multi-label learning