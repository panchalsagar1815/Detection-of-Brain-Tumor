# Detection-of-Brain-Tumor
**Project: Detection of Brain Tumor**

**Objective:**
The "Detection of Brain Tumor" project aims to develop a robust classification system to identify brain tumours from medical images. The dataset consists of images categorized into 'no_tumor,' 'pituitary_tumor,' 'meningioma_tumor,' and 'glioma_tumor.' Through comprehensive data preprocessing, cleaning, augmentation, and the implementation of a DenseNet201 model, the project endeavours to accurately classify brain tumours into their respective categories.

**Data Processing:**
1. **Data Cleaning and Preprocessing:**
   - Addressed image artefacts, noise, and inconsistencies for reliable model training.
   - Ensured uniformity and quality in the dataset.

2. **Data Visualization and EDA:**
   - Explored image distributions, pixel intensities, and tumour patterns.
   - Gained insights into the characteristics of different tumour types.

**Machine Learning Model:**
Utilized the DenseNet201 model for tumour classification:
```python
inp = model1.input
''' Hidden Layer '''
x = tf.keras.layers.Dense(128, activation='relu')(model1.output)
''' Classification Layer '''
out = tf.keras.layers.Dense(4, activation='softmax')(x)

''' Model '''
model = tf. keras.Model(inputs=inp, outputs=out)

''' Compile the Model '''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

**Model Features:**
- **DenseNet201:**
  - Leveraged for its deep architecture and ability to capture intricate features.
  - Employed transfer learning to benefit from pre-trained weights on large datasets.

**Data Augmentation:**
Applied techniques to enhance the dataset's diversity and robustness:
- Image reshaping and resizing.
- Label encoding for categorical classification.
- Augmented data with rotations, flips, and zooms for improved model generalization.

**Precision and Recall Calculation:**
Evaluation metrics such as precision and recall were calculated to assess the model's ability to correctly classify brain tumors into their respective categories. Precision measures the accuracy of positive predictions, while recall assesses the model's coverage of actual positive instances.

**Conclusion:**
The "Detection of Brain Tumor" project successfully implemented a DenseNet201 model for accurate classification of brain tumors. Leveraging advanced techniques such as data augmentation, label encoding, and transfer learning, the model demonstrates promising results in identifying 'no_tumor,' 'pituitary_tumor,' 'meningioma_tumor,' and 'glioma_tumor.' The precision and recall metrics provide a comprehensive understanding of the model's performance, contributing to its reliability in real-world medical applications.
