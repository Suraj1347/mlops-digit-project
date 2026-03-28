from sklearn import datasets, svm
import pickle

print("Loading dataset...")
digits = datasets.load_digits() # Built-in 8x8 pixel digit dataset

# Flatten the images and train a lightweight Support Vector Machine
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("Model trained and saved as model.pkl!")