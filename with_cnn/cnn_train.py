import os
import argparse
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class MultiViewDataset(Dataset):
    """
    PyTorch Dataset for multi-view biomedical images.
    Each sample has three views (axial, coronal, sagittal).
    We implement early fusion by stacking the three grayscale views as channels.
    """
    def __init__(self, df, image_dir, transform=None):
        """
        df: DataFrame with columns ['id', 'class'].
        image_dir: root directory containing subfolders 'train/' with images named a_*.png, c_*.png, s_*.png.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        # Create mapping from class label to index
        classes = sorted(df['class'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load base name (strip 'a_','c_','s_' prefix)
        img_id = self.df.loc[idx, 'id']
        label = self.df.loc[idx, 'class']
        base = img_id.split('_', 1)[1]
        # Paths for axial, coronal, sagittal images
        paths = [
            os.path.join(self.image_dir, 'train', f'a_{base}'),
            os.path.join(self.image_dir, 'train', f'c_{base}'),
            os.path.join(self.image_dir, 'train', f's_{base}')
        ]
        images = []
        for path in paths:
            if os.path.exists(path):
                img = Image.open(path).convert('L')
            else:
                # If a view is missing, substitute a blank image
                img = Image.new('L', (128, 128))
            images.append(img)
        # Merge into one 3-channel image: R=axial, G=coronal, B=sagittal
        multi = Image.merge('RGB', (images[0], images[1], images[2]))
        if self.transform:
            multi = self.transform(multi)
        label_idx = self.class_to_idx[label]
        return multi, label_idx

def train(args):
    # Load cleaned training IDs and labels
    if args.train_pickle and os.path.exists(args.train_pickle):
        df = pd.read_pickle(args.train_pickle)
    else:
        df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    # Drop duplicate bases so each volume counts once
    df['base'] = df['id'].apply(lambda x: x.split('_',1)[1])
    df = df.drop_duplicates(subset='base').drop(columns=['base'])
    # Shuffle and split into training/validation
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = int(len(df) * args.val_frac)
    df_val = df[:n_val]
    df_train = df[n_val:]
    # Data augmentations for training
    transform_train = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])
    train_dataset = MultiViewDataset(df_train, args.data_dir, transform=transform_train)
    val_dataset   = MultiViewDataset(df_val,   args.data_dir, transform=transform_val)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Save class mapping (label->index) for inference later
    class_map = train_dataset.class_to_idx
    torch.save(class_map, args.output_prefix + '_class_map.pth')
    
    # A simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),  # input channels = 3
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128 * 8 * 8, 256),  # input 128x128 -> 8x8 after 4 pools
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(class_map)).to(device)
    print(f"Using device: {device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_val_acc = 0.0
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = total = 0
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validating]"):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={epoch_loss:.4f} ValAcc={val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_prefix + '_best.pth')
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-view classification CNN')
    parser.add_argument('--data_dir', type=str, default='organ-all-mnist-DL20242025',
                        help='Root dataset directory containing train/ and test/ folders')
    parser.add_argument('--train_pickle', type=str, default='train_clean.pkl',
                        help='Pickle of valid train IDs (from preprocessing)')
    parser.add_argument('--output_prefix', type=str, default='cnn',
                        help='Prefix for saving model and class map files')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--val_frac', type=float, default=0.2, help='Fraction of data for validation')
    args = parser.parse_args()
    train(args)
