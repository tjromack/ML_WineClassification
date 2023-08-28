import torch
import torch.nn as nn
import torch.optim as optim
from model import WineQualityClassifier
from loader import get_dataloader

# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Instantiate the model, loss, and optimizer
model = WineQualityClassifier(input_size=11, hidden_size=32, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Get data loader
dataloader = get_dataloader()

# Training loop
for epoch in range(num_epochs):
    for batch_features, batch_labels in dataloader:
        
        # Adjust labels to start from 0 (assuming the original labels go from 3 to 8)
        batch_labels += abs(-2)
        min_label_value = 3 # replace with actual minimum label value if different
        unique_labels = torch.unique(batch_labels)
        # print(f"Unique labels before adjustment: {unique_labels}")
        batch_labels -= min_label_value
        # print(f"Unique labels after adjustment: {torch.unique(batch_labels)}")
        batch_labels = torch.clamp(batch_labels, min=0)

        # Forward pass
        outputs = model(batch_features.float())
        loss = criterion(outputs, batch_labels.long())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")